# Codes modified from https://github.com/HanGuo97/lq-lora/blob/main/models/quantization_utils.py

import math
import torch
import scipy.special

import torch.nn as nn

from typing import Tuple, NamedTuple


NF4_OFFSET = 0.9677083  # Magic number?


class NFQuantizedWeights(torch.nn.Module):
    def __init__(self, weight, qscheme, scale, dtype=torch.float32):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        
        self.dtype = dtype

        weight_q = nf_quant(weight, qscheme, scale)
            
        self.scale = nn.Parameter(scale)
        self.qscheme = qscheme
            
        self.register_buffer("weight_q", weight_q)

    def forward(self):
        return nf_dequant(self.weight_q, self.qscheme, self.scale).to(self.dtype)


class QuantScheme(NamedTuple):
    values: torch.Tensor
    boundaries: torch.Tensor


def create_quantization_scheme(
    values: torch.Tensor,
    device: torch.device,
) -> QuantScheme:
    inf_tensor = torch.tensor([torch.inf])
    boundaries = (values[1:] + values[:-1]) / 2.
    boundaries = torch.cat([-inf_tensor, boundaries, inf_tensor], dim=0)

    values = values.to(device=device)
    boundaries = boundaries.to(device=device)
    if values.ndim != 1 or boundaries.ndim != 1:
        raise ValueError
    if values.shape[0] != boundaries.shape[0] - 1:
        raise ValueError
    return QuantScheme(
        values=values,
        boundaries=boundaries)
    
    
def quantize_with_scheme(
    A: torch.Tensor,
    qscheme: QuantScheme,
    scales_q: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # if A.shape != scales_q.shape:
    #     raise ValueError
    A_scaled = A / scales_q
    # `-1` because this function assigns to the right bucket
    A_quantized = torch.bucketize(
        A_scaled,
        qscheme.boundaries,
        right=False) - 1
    A_dequantized = qscheme.values[A_quantized] * scales_q
    return A_quantized, A_dequantized


def create_normal_float_scheme(
    num_bits: int,
    device: torch.device,
) -> QuantScheme:
    # This is essentially what NF4 does.
    sigma = -1. / (
        math.sqrt(2) *
        scipy.special.erfinv(1 - 2 * NF4_OFFSET))
    qdist = torch.distributions.normal.Normal(
        loc=0.,
        scale=sigma)

    quantiles_left = torch.linspace(
        1. - NF4_OFFSET,
        0.5,
        2 ** (num_bits - 1))
    quantiles_right = torch.linspace(
        0.5,
        NF4_OFFSET,
        2 ** (num_bits - 1) + 1)
    # remove the duplicated `0.5`
    quantiles = torch.cat([
        quantiles_left[:-1],
        quantiles_right],
        dim=0)
    values = qdist.icdf(quantiles)
    return create_quantization_scheme(
        values=values,
        device=device)
    

def nf_quant(x, qscheme, scale):
    scale = scale.to(x.device)
    x_scaled = x / scale
    # `-1` because this function assigns to the right bucket
    q = torch.bucketize(
        x_scaled,
        qscheme.boundaries.to(x.device),
        right=False) - 1
    return q


def nf_dequant(q, qscheme, scale):
    return qscheme.values.to(q.device)[q] * scale.to(q.device)


def nf_quant_dequant(x, qscheme, scale):
    return nf_dequant(nf_quant(x, qscheme, scale), qscheme, scale)
    

if __name__ == "__main__":
    x = torch.randn(2, 3)
    
    dev = x.device
    
    tmp = torch.zeros(x.shape[0], device=dev)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)
    
    scheme = create_normal_float_scheme(3, x.device)

    xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
    print(scheme.values)
    # print(abs(scheme.values[0]), scheme.values[-1])
    grid_max = max(abs(scheme.values[0]), scheme.values[-1])
    scale = xmax / grid_max
    
    print(x)
    
    print(quantize_with_scheme(x, scheme, scale.unsqueeze(1))[1])
    print(nf_quant_dequant(x, scheme, scale.unsqueeze(1)))
    
    assert torch.all(nf_quant_dequant(x, scheme, scale.unsqueeze(1)) == quantize_with_scheme(x, scheme, scale.unsqueeze(1))[1])
import torch
import math
import time
import logging
import utils

import math
from functools import cache

import quiptools_cuda
import torch
from torch import nn

import torch
import gptq_utils


_E8P_CODESZ = 8

_E8P_SCALE = 1.03


def get_norm12():
    # 29 elements of norm 12 in E8 + 1/4
    return torch.tensor([
        [3, 1, 1, 1, 3, 3, 3, 3],
        [1, 3, 1, 1, 3, 3, 3, 3],
        [1, 1, 3, 1, 3, 3, 3, 3],
        [1, 1, 1, 3, 3, 3, 3, 3],
        [3, 3, 3, 1, 3, 3, 1, 1],
        [3, 3, 3, 1, 3, 1, 3, 1],
        [3, 3, 3, 1, 1, 3, 3, 1],
        [3, 3, 3, 1, 3, 1, 1, 3],
        [3, 3, 3, 1, 1, 3, 1, 3],
        [3, 3, 3, 1, 1, 1, 3, 3],
        [3, 3, 1, 3, 3, 3, 1, 1],
        [3, 3, 1, 3, 3, 1, 3, 1],
        [3, 3, 1, 3, 1, 3, 3, 1],
        [3, 3, 1, 3, 3, 1, 1, 3],
        [3, 3, 1, 3, 1, 3, 1, 3],
        [3, 3, 1, 3, 1, 1, 3, 3],
        [3, 1, 3, 3, 3, 3, 1, 1],
        [3, 1, 3, 3, 3, 1, 3, 1],
        [3, 1, 3, 3, 1, 3, 3, 1],
        [3, 1, 3, 3, 3, 1, 1, 3],
        [3, 1, 3, 3, 1, 3, 1, 3],
        [1, 3, 3, 3, 1, 1, 3, 3],
        [1, 3, 3, 3, 3, 3, 1, 1],
        [1, 3, 3, 3, 3, 1, 3, 1],
        [1, 3, 3, 3, 1, 3, 3, 1],
        [1, 3, 3, 3, 3, 1, 1, 3],
        [1, 3, 3, 3, 1, 3, 1, 3],
        [1, 1, 3, 3, 1, 3, 3, 3],
        [3, 3, 1, 1, 3, 3, 3, 1],
    ]) / 2


def get_packed_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * 8).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1)**2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)
    cba = cba[:, [0, 2, 4, 6, 1, 3, 5, 7]]
    cba[:, 7] *= (1 - 2 * (cba.sum(1) % 2))
    cba = cba * 2 + 8
    cba = cba.to(torch.int32)
    acc = cba[:, 0]
    for i in range(7):
        acc = acc | (cba[:, (i + 1)] << ((i + 1) * 4))
    return acc


def get_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * _E8P_CODESZ).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1)**2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)
    return cba


def get_full_grid(packed_abs_grid):
    synth_codebook = torch.zeros(1 << 16, 8)
    parity_idx = []
    shuffle_map = [0, 4, 1, 5, 2, 6, 3, 7]
    for c in range(1 << 16):
        signs = c & 255
        abs = c >> 8
        parity = 0
        for i in range(8):
            parity = parity ^ ((signs >> i) & 1)
        signs = signs ^ parity
        abs_code = packed_abs_grid[abs].item()
        for i in range(8):
            ii = shuffle_map[i]
            synth_codebook[c, i] = (((abs_code >> (4 * ii)) & 15) - 8) * 0.5
            if ((signs >> ii) & 1):
                synth_codebook[c, i] *= -1
        if parity:
            synth_codebook[c, :] -= 0.25
            parity_idx.append(c)
        else:
            synth_codebook[c, :] += 0.25
    return synth_codebook, torch.arange(1 << 16), parity_idx


_E8P_PACKED_ABS_CACHED = get_packed_abs_grid()
_E8P_GRID, _E8P_GRID_IDX, _PARITY_IDX = get_full_grid(_E8P_PACKED_ABS_CACHED)


def block_LDL(H, b, check_nan=True, add_until_fail=True, percdamp=.01):
    n = H.shape[0]
    assert (n % b == 0)
    m = n // b
    
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[1], device=H.device)
    
    if add_until_fail:
        multiplier = 1
        
        while multiplier < 50:
            try:
                H[diag, diag] += damp
                L = torch.linalg.cholesky(H)
                break
            except:
                multiplier += 1
    else:
        try:
            L = torch.linalg.cholesky(H)
        except:
            return None
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    D = (DL @ DL.permute(0, 2, 1)).cpu()
    DL = torch.linalg.inv(DL)
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]

    if check_nan and L.isnan().any():
        return None

    L = L.reshape(n, n)
    return (L, D.to(DL.device))


class LDLQ():

    def __init__(
        self, 
        layer, 
        normalize_over_tokens=False, 
        normalize_hessian=False, 
        add_until_fail=False,
        low_rank_before=False,
        low_rank_after=False,
        rank_ratio=1.0,
        rank_take_top=False,
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.normalize_over_tokens = normalize_over_tokens
        self.normalize_hessian = normalize_hessian
        self.add_until_fail = add_until_fail
        self.low_rank_before = low_rank_before
        self.low_rank_after = low_rank_after
        self.rank_ratio = rank_ratio
        self.rank_take_top = rank_take_top
        
        self.idx_dtype = torch.int32
        
        self.codesz = _E8P_CODESZ
        
        self.grid = _E8P_GRID
        self.grid_norm = _E8P_GRID.norm(dim=-1)**2
        grid_part = _E8P_GRID[_PARITY_IDX] + 0.25
        grid_part = grid_part[
            torch.where(
                ((grid_part[:, :7] < 0).sum(dim=-1) <= 1) * \
                (grid_part[:, :7].min(dim=-1).values >= -0.5)
            )[0]]
        self.grid_part = grid_part
        self.grid_part_norm = grid_part.norm(dim=-1)**2
        
        abs_grid = get_abs_grid()
        self.grid_abs_odd = abs_grid.sum(dim=-1) % 2 == 1
        
        self.part_abs_map = self.round(grid_part.abs(), abs_grid, abs_grid.norm(dim=-1)**2)[1]
        self.bit_map = 2**torch.arange(8)
    
        self.grid = self.grid.to(self.dev)
        self.grid_norm = self.grid_norm.to(self.dev)
        self.grid_part = self.grid_part.to(self.dev)
        self.grid_part_norm = self.grid_part_norm.to(self.dev)
        self.grid_abs_odd = self.grid_abs_odd.to(self.dev)
        self.part_abs_map = self.part_abs_map.to(self.dev)
        self.bit_map = self.bit_map.to(self.dev)

    def add_batch(self, inp, out, weighting=None, feature_weighting=None, sequence_weighting=None):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()

        if weighting is not None:
            # normalize weighting
            weighting = weighting / weighting.sum() * weighting.shape[0]
            inp = inp * weighting.to(inp.device) ** 0.5
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        
        if feature_weighting is not None:
            feature_weighting = feature_weighting / feature_weighting.sum() * feature_weighting.shape[0]
            inp = feature_weighting.to(inp.device)[:, None] ** 0.5 * inp
            
        if sequence_weighting is not None:
            inp = sequence_weighting.to(inp.device) ** 0.5 * inp

        if self.normalize_over_tokens:
            inp = inp / inp.shape[-1] ** 0.5

        self.H += inp.matmul(inp.t())
        
    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx
        
    def fast_quantize_part(self, X, parity):
        X_part = torch.abs(X)
        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 7] = -X_part[X_odd, 7]
        mask = 1 - 2 * (X < 0).to(torch.float32)
        mask[X_odd, 7] = -mask[X_odd, 7]
        roundout, Xqidx = self.round(X_part, self.grid_part,
                                     self.grid_part_norm)
        vals = roundout * mask
        err = (X - vals).norm(dim=-1)
        abs_idx = self.part_abs_map[Xqidx]
        sign_mask = (((roundout < 0) ^ (mask < 0))[:,
                                                   [0, 2, 4, 6, 1, 3, 5, 7]])
        sign_mask[:, 7] = sign_mask[:, 7] ^ self.grid_abs_odd[abs_idx]
        sign_mask[:, 0] = sign_mask[:, 0] ^ parity
        mask_idx = (sign_mask * self.bit_map).sum(dim=-1).int()
        idx = (abs_idx << 8) + mask_idx
        return vals, idx, err
    
    def quantize_piece(self, x, **kwargs):
        # find the quantized val and index, used for adaptive rounding
        X_plus = x + 1 / 4  # quantize X to D8^ - 1/4
        X_minus = x - 1 / 4  # quantize X to D8^ + 1/4

        plus_vals, plus_idx, plus_err = self.fast_quantize_part(X_plus, True)
        minus_vals, minus_idx, minus_err = self.fast_quantize_part(
            X_minus, False)

        which = plus_err < minus_err
        final_vals = torch.where(which.unsqueeze(-1), plus_vals - 1 / 4,
                                 minus_vals + 1 / 4)
        final_idx = torch.where(which, plus_idx, minus_idx)

        return final_vals, final_idx
    
    def LDLQ(self, Wr, Hr, blocksize, resid_scale_override=-1, quip_tune_iters=10):
        '''
        want eta = (Wr - hatWr) @ L
        want hatWr + eta = Wr + (Wr - hatWr) @ (L - I)
        want hatWr = Q( Wr + (Wr - hatWr) @ (L - I) )
        '''
        L, D = block_LDL(Hr, blocksize, add_until_fail=self.add_until_fail)

        (m, n) = Wr.shape
        
        hatWr = torch.zeros(m, n, dtype=Hr.dtype, device=Hr.device)
        Qidxs = torch.zeros(m,
                            n // blocksize,
                            dtype=self.idx_dtype,
                            device=Hr.device)
        # import pdb; pdb.set_trace()
        # norms = []
        for k in reversed(range(n // blocksize)):
            WXWX = Wr[:, (blocksize * k):(blocksize * (k + 1))] + \
                (Wr[:, (blocksize * (k + 1)):n] - hatWr[:, (blocksize * (k + 1)):n]) @ \
                L[(blocksize * (k + 1)):n, (blocksize * k):(blocksize * (k + 1))]
            
            hatWr[:, (blocksize * k):(blocksize * (k + 1))], Qidxs[:, k] = \
                self.quantize_piece(WXWX, resid_scale_override=resid_scale_override, index=blocksize*k)
            
            # if torch.any(WXWX.norm(dim=-1) > 3):
            #     import pdb; pdb.set_trace()
            # norms.append(WXWX.norm(dim=-1))
            
        for ie in range(quip_tune_iters):
            for k in reversed(range(n // blocksize)):
                WXWX = hatWr[:, (blocksize * k):(blocksize * (k + 1))] + (Wr - hatWr) @ \
                    Hr[:, (blocksize * k):(blocksize * (k + 1))] @ \
                    torch.linalg.inv(Hr[(blocksize * k):(blocksize * (k + 1)),
                                        (blocksize * k):(blocksize * (k + 1))])
                hatWr[:, (blocksize *
                        k):(blocksize * (k + 1))], Qidxs[:, k] = self.quantize_piece(
                            WXWX, resid_scale_override=resid_scale_override, index=blocksize*k)

        return hatWr, Qidxs
    
    def quantize(self, x, H, quip_tune_iters=10, resid_scale_override=-1):
        _, q = self.LDLQ(
            x, H, self.codesz,
            resid_scale_override=resid_scale_override, quip_tune_iters=quip_tune_iters,
        )
            
        return q

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, quant=True
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)
            
        if not quant:
            return

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Q = self.quantize(W / self.quantizer.scale, H, quip_tune_iters=10, resid_scale_override=-1) # default setup
        
        quantized_weight = E8PQuantizedWeights(
            Q, self.quantizer.scale, self.grid, W.shape[0], W.shape[1], dtype=self.layer.weight.data.dtype
        ).to(self.dev)
        
        self.quantizer.quantized_weight = quantized_weight
        
        deQ = quantized_weight.forward()
        
        torch.cuda.synchronize()
            
        self.layer.weight.data = deQ.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')
        
    def get_quantize_linear(self, qat=False):
        quantized_weight = self.quantizer.quantize(
            self.layer.weight.data,
            qat,
        )
        
        return gptq_utils.QuantizedLinear(quantized_weight, self.layer.bias)

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


class E8PQuantizedWeights(torch.nn.Module):
    def __init__(self, weight_q, scale, grid, out_features, in_features, dtype=torch.float32, **kwargs):
        super().__init__()
        self.out_features, self.in_features = out_features, in_features
        
        self.codesz = _E8P_CODESZ
        self.register_buffer('grid', grid)
        
        self.dtype = dtype
            
        self.scale = nn.Parameter(scale)
        self.register_buffer("weight_q", weight_q)

    def forward(self):
        return self.dequantize(self.weight_q, self.scale).to(self.dtype)

    def dequantize(self, quantized_x, scale, **kwargs):
        return self.grid[quantized_x.int()].reshape(self.out_features, self.in_features) * scale


class E8PWeightQuantizer(torch.nn.Module):
    def __init__(self, shape=1):
        super(E8PWeightQuantizer, self).__init__()
        self.register_buffer('scale', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8, scale_override=0.9, **kwargs,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        
        # took from quip_sharp project
        # https://github.com/Cornell-RelaxML/quip-sharp/blob/main/quantize_llama/example.sh
        self.scale_override = scale_override

    def find_params(self, x):
        if self.bits == 16:
            return
        
        p = 2
        
        scale = x.norm(p=p)
        scale /= x.numel() ** 0.5
        
        if self.scale_override > 0:
            self.scale = scale / self.scale_override
        else:
            # use E8P const to rescale
            self.scale = scale / _E8P_SCALE
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def forward(self, x):
        raise NotImplementedError
    
    def quantize(self, x, qat=True):
        if qat:
            raise NotImplementedError

        assert getattr(self, "quantized_weight", None) is not None, "the quantized weight is not set: quantize with the LDLQ first"
        return self.quantized_weight

    def ready(self):
        return torch.all(self.scale != 0)

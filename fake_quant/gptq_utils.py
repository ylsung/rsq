import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from transformers import PreTrainedModel

from tqdm import trange
from tqdm.auto import trange

import utils
import copy
import quant_utils
import logging
import schedulers
import input_weighting_module
import optimizers
import attn_module

from model_utils import (
    FALCON_TYPES,
    get_layers,
)
import ldlq_utils
from collections import defaultdict


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def low_rank_approximation(matrix, rank, take_top=True):
    """
    Returns the low-rank approximation of the input matrix using its top 'rank' singular values and vectors.

    Parameters:
    matrix (torch.Tensor): The input 2D tensor to be approximated.
    rank (int): The number of top singular values and vectors to use for the approximation.

    Returns:
    torch.Tensor: The low-rank approximation of the input matrix.
    """
    # Perform Singular Value Decomposition
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

    # Select the top 'rank' singular values and corresponding vectors
    if take_top:
        U_k = U[:, :rank]
        S_k = S[:rank]
        Vh_k = Vh[:rank, :]
    else:
        U_k = U[:, -rank:]
        S_k = S[-rank:]
        Vh_k = Vh[-rank:, :]
    
    # Reconstruct the low-rank approximation
    low_rank_matrix = U_k @ torch.diag(S_k) @ Vh_k

    return low_rank_matrix


class QuantizedLinear(nn.Module):
    # modified from https://github.com/Vahe1994/AQLM/blob/a441a3f0ece4cbaa2a91a3421c95a8b7432e4d99/src/aq.py#L18C1-L34C36
    def __init__(self, quantized_weight, bias):
        super().__init__()
        self.out_features, self.in_features = quantized_weight.out_features, quantized_weight.in_features
        self.quantized_weight = quantized_weight
        self.bias = bias
        self.use_checkpoint = False

    def _forward(self, input: torch.Tensor):
        return F.linear(input, self.quantized_weight(), self.bias)

    def forward(self, input: torch.Tensor):
        if getattr(self, "use_checkpoint", False) and torch.is_grad_enabled():
            return checkpoint(
                self._forward, input, use_reentrant=False, preserve_rng_state=False, determinism_check="none"
            )
        return self._forward(input)

    def to_fake_quant_linear(self):
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        linear.weight.data = self.quantized_weight()
        if self.bias is not None:
            linear.bias.data = self.bias
        
        return linear


class GPTQ:

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

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        
        if self.low_rank_before and self.rank_ratio < 1.0:
            rank_to_use = int(self.rank_ratio * H.shape[0])
            H = low_rank_approximation(H, rank_to_use, take_top=True)

        if self.normalize_hessian:
            H.div_(torch.mean(torch.diag(H)))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += percdamp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
        else:
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            
            if self.add_until_fail:
                multiplier = 1
                
                while multiplier < 50:
                    try:
                        H[diag, diag] += damp
                        H = torch.linalg.cholesky(H)
                        H = torch.cholesky_inverse(H)
                        H = torch.linalg.cholesky(H, upper=True)
                        break
                    except:
                        multiplier += 1
                        
                print(multiplier)
            else:
                H[diag, diag] += damp
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                
        if self.low_rank_after and self.rank_ratio < 1.0:
            rank_to_use = int(self.rank_ratio * H.shape[0])
            H = low_rank_approximation(H, rank_to_use, take_top=self.rank_take_top)
            
        # except:
            # import pdb; pdb.set_trace()
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.forward(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]
            
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')
        
    def get_quantize_linear(self, qat=True):
        quantized_weight = self.quantizer.quantize(
            self.layer.weight.data,
            qat,
        )
        
        return QuantizedLinear(quantized_weight, self.layer.bias)

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


class AdaptiveGPTQ(GPTQ):
    def __init__(self, layer, normalize_over_tokens=False, values_to_try=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3], train_in_val=False):
        super().__init__(layer, normalize_over_tokens)
        self.H_val = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples_val = 0
        self.values_to_try = values_to_try
        self.values_to_try.insert(0, 0.0)
        self.train_in_val = train_in_val

    def add_batch_val(self, inp, out, weighting=None, feature_weighting=None):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H_val *= self.nsamples_val / (self.nsamples_val + tmp)
        self.nsamples_val += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples_val) * inp.float()

        if weighting is not None:
            # normalize weighting
            weighting = weighting / weighting.sum() * weighting.shape[0]
            inp = inp * weighting.to(inp.device) ** 0.5
            
        if feature_weighting is not None:
            inp = inp * feature_weighting.to(inp.device) ** 0.5

        if self.normalize_over_tokens:
            inp = inp / inp.shape[-1] ** 0.5

        self.H_val += inp.matmul(inp.t())
        
    def _channelwise_squared_error(self, XTX: torch.Tensor, weight: torch.Tensor, reference_weight: torch.Tensor):
        """
        Compute per-channel squared error between X @ weight_or_weights and X @ reference_weight
        :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
        :note: if XTX is divided by dataset size, this function will return *mean* squared error
        :param weight: predicted/reconstructed weights of shape [*dims, out_features, in_features]
        :param reference_weight: reference weight of shape [out_features, in_features]
        :return: per-channel squared errors of shape [*dims, out_features]
        """
        XW_norm_square = torch.matmul(weight[..., :, None, :], (weight @ XTX)[..., :, :, None]).flatten(-3)
        XWreference_norm_square = torch.bmm(reference_weight[:, None, :], (reference_weight @ XTX)[:, :, None]).flatten(-3)
        dot_product = torch.matmul((reference_weight @ XTX)[:, None, :], weight[..., :, :, None]).flatten(-3)
        return XW_norm_square - 2 * dot_product + XWreference_norm_square

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
        
        values_to_try = self.values_to_try
        
        val_losses = []
        
        original_H = self.H.to("cpu")
        original_W = W.to("cpu")
        
        del self.H

        for value in values_to_try:
            H = original_H.clone().to(self.dev)
            W = original_W.clone().to(self.dev)
            Losses = torch.zeros_like(W)
            Q = torch.zeros_like(W)
            
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0

            if static_groups:
                import copy
                groups = []
                for i in range(0, self.columns, groupsize):
                    quantizer = copy.deepcopy(self.quantizer)
                    quantizer.find_params(W[:, i:(i + groupsize)])
                    groups.append(quantizer)

            if actorder:
                perm = torch.argsort(torch.diag(H), descending=True)
                W = W[:, perm]
                H = H[perm][:, perm]
                invperm = torch.argsort(perm)
            
            diag = torch.arange(self.columns, device=self.dev)

            # print(H.abs().mean(), W.abs().mean()) # should be the same for all values

            try:
                damp = value * torch.mean(torch.diag(H))
                H[diag, diag] += damp
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                # except:
                    # import pdb; pdb.set_trace()
                Hinv = H
            except:
                val_losses.append(float("inf"))
                # cannot get H inv, don't need to do anything more
                # print(f"val: {value}, val_loss: {val_losses[-1]}")
                continue

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if groupsize != -1:
                        if not static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                        else:
                            idx = i1 + i
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // groupsize]

                    q = self.quantizer.forward(w.unsqueeze(1)).flatten()
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                Losses[:, i1:i2] = Losses1 / 2

                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
                
            if actorder:
                Q = Q[:, invperm]
            
            if self.train_in_val:
                val_loss = self._channelwise_squared_error(
                    0.5 * self.H_val + 0.5 * original_H.to(self.dev), 
                    Q,
                    original_W.to(self.dev),
                )
            else:
                val_loss = self._channelwise_squared_error(
                    self.H_val, 
                    Q, 
                    original_W.to(self.dev),
                )
            
            val_loss = val_loss.sum()
            val_losses.append(val_loss.item())
            
            # print(f"val: {value}, val_loss: {val_losses[-1]} Q: {Q.to(self.layer.weight.data.dtype).abs().mean().item()}")
            
        del W
        del H
        del Q
        
        percdamp = values_to_try[val_losses.index(min(val_losses))]
        logging.info(f"values: {values_to_try}")
        logging.info(f"losses: {val_losses}, best_perdamp: {percdamp}")
        
        self.H = original_H.clone().to(self.dev)
        return super().fasterquant(
            blocksize=blocksize, percdamp=percdamp, groupsize=groupsize, actorder=actorder, static_groups=static_groups
        )

    def free(self):
        self.H = None
        self.H_val = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)
        
        
class AdaptiveLowRankGPTQ(GPTQ):
    def __init__(self, 
        layer, 
        normalize_over_tokens=False, 
        values_to_try=[0.95, 0.9, 0.7], 
        train_in_val=False,
        low_rank_before=False,
        low_rank_after=False,
        rank_ratio=1.0,
        rank_take_top=False,
    ):
        super().__init__(
            layer, 
            normalize_over_tokens,
            low_rank_before=low_rank_before,
            low_rank_after=low_rank_after,
            rank_ratio=rank_ratio,
            rank_take_top=rank_take_top,
        )
        self.H_val = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples_val = 0
        self.values_to_try = values_to_try
        self.values_to_try.insert(0, 1.0)
        self.train_in_val = train_in_val

    def add_batch_val(self, inp, out, weighting=None):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H_val *= self.nsamples_val / (self.nsamples_val + tmp)
        self.nsamples_val += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples_val) * inp.float()

        if weighting is not None:
            # normalize weighting
            weighting = weighting / weighting.sum() * weighting.shape[0]
            inp = inp * weighting.to(inp.device) ** 0.5

        if self.normalize_over_tokens:
            inp = inp / inp.shape[-1] ** 0.5

        self.H_val += inp.matmul(inp.t())
        
    def _channelwise_squared_error(self, XTX: torch.Tensor, weight: torch.Tensor, reference_weight: torch.Tensor):
        """
        Compute per-channel squared error between X @ weight_or_weights and X @ reference_weight
        :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
        :note: if XTX is divided by dataset size, this function will return *mean* squared error
        :param weight: predicted/reconstructed weights of shape [*dims, out_features, in_features]
        :param reference_weight: reference weight of shape [out_features, in_features]
        :return: per-channel squared errors of shape [*dims, out_features]
        """
        XW_norm_square = torch.matmul(weight[..., :, None, :], (weight @ XTX)[..., :, :, None]).flatten(-3)
        XWreference_norm_square = torch.bmm(reference_weight[:, None, :], (reference_weight @ XTX)[:, :, None]).flatten(-3)
        dot_product = torch.matmul((reference_weight @ XTX)[:, None, :], weight[..., :, :, None]).flatten(-3)
        return XW_norm_square - 2 * dot_product + XWreference_norm_square

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
        
        values_to_try = self.values_to_try
        
        val_losses = []
        
        original_H = self.H.to("cpu")
        original_W = W.to("cpu")
        
        del self.H

        for value in values_to_try:
            H = original_H.clone().to(self.dev)
            W = original_W.clone().to(self.dev)
            Losses = torch.zeros_like(W)
            Q = torch.zeros_like(W)
            
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0

            if static_groups:
                import copy
                groups = []
                for i in range(0, self.columns, groupsize):
                    quantizer = copy.deepcopy(self.quantizer)
                    quantizer.find_params(W[:, i:(i + groupsize)])
                    groups.append(quantizer)

            if actorder:
                perm = torch.argsort(torch.diag(H), descending=True)
                W = W[:, perm]
                H = H[perm][:, perm]
                invperm = torch.argsort(perm)
            
            diag = torch.arange(self.columns, device=self.dev)

            # print(H.abs().mean(), W.abs().mean()) # should be the same for all values
            
            if self.low_rank_before and value < 1.0:
                rank_to_use = int(value * H.shape[0])
                H = low_rank_approximation(H, rank_to_use, take_top=True)

            try:
                damp = percdamp * torch.mean(torch.diag(H))
                H[diag, diag] += damp
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                # except:
                    # import pdb; pdb.set_trace()
            except:
                val_losses.append(float("inf"))
                # cannot get H inv, don't need to do anything more
                # print(f"val: {value}, val_loss: {val_losses[-1]}")
                continue
            
            if self.low_rank_after and value < 1.0:
                rank_to_use = int(value * H.shape[0])
                H = low_rank_approximation(H, rank_to_use, take_top=self.rank_take_top)
            
            Hinv = H

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if groupsize != -1:
                        if not static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                        else:
                            idx = i1 + i
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // groupsize]

                    q = self.quantizer.forward(w.unsqueeze(1)).flatten()
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                Losses[:, i1:i2] = Losses1 / 2

                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
                
            if actorder:
                Q = Q[:, invperm]
            
            if self.train_in_val:
                val_loss = self._channelwise_squared_error(
                    0.5 * self.H_val + 0.5 * original_H.to(self.dev), 
                    Q,
                    original_W.to(self.dev),
                )
            else:
                val_loss = self._channelwise_squared_error(
                    self.H_val, 
                    Q, 
                    original_W.to(self.dev),
                )
            
            val_loss = val_loss.sum()
            val_losses.append(val_loss.item())
            
            # print(f"val: {value}, val_loss: {val_losses[-1]} Q: {Q.to(self.layer.weight.data.dtype).abs().mean().item()}")
            
        del W
        del H
        del Q
        
        rank_ratio = values_to_try[val_losses.index(min(val_losses))]
        logging.info(f"values: {values_to_try}")
        logging.info(f"losses: {val_losses}, best_rank_ratio: {rank_ratio}")
        
        self.H = original_H.clone().to(self.dev)
        self.rank_ratio = rank_ratio
        return super().fasterquant(
            blocksize=blocksize, percdamp=percdamp, groupsize=groupsize, actorder=actorder, static_groups=static_groups
        )

    def free(self):
        self.H = None
        self.H_val = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


class GPTQ_sim(GPTQ):

    def __init__(self, layer, alpha=1):
        super().__init__(layer)
        self.alpha = alpha

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
        
        # original H (2XX^T) + H for the similarity term (8 XX^T W^TW XX^T)
        # note that self.H = 2XX^T
        H = H + 2 * self.alpha * H.matmul(W.t()).matmul(W).matmul(H)

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.forward(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]
            
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')
        


class GPTQ_TP(GPTQ):

    def __init__(self, layer):
        super().__init__(layer)
        self.cached_clean_input = None
        self.cached_distorted_input = None
        self.H_star = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples_star = 0
        
    def set_clean_input(self, inp):
        self.cached_clean_input = inp
    
    def set_distorted_input(self, inp):
        self.cached_distorted_input = inp
        
    def add_batch_for_G(self):
        
        assert self.cached_clean_input != None and self.cached_distorted_input != None, "Inputs are not set"
        
        cached_clean_input = self.cached_clean_input
        cached_distorted_input = self.cached_distorted_input
        
        if len(cached_clean_input.shape) == 2:
            cached_clean_input = cached_clean_input.unsqueeze(0)
        tmp = cached_clean_input.shape[0]
        if len(cached_clean_input.shape) == 3:
            cached_clean_input = cached_clean_input.reshape((-1, cached_clean_input.shape[-1]))
        cached_clean_input = cached_clean_input.t()
        
        if len(cached_distorted_input.shape) == 2:
            cached_distorted_input = cached_distorted_input.unsqueeze(0)
        tmp = cached_distorted_input.shape[0]
        if len(cached_distorted_input.shape) == 3:
            cached_distorted_input = cached_distorted_input.reshape((-1, cached_distorted_input.shape[-1]))
        cached_distorted_input = cached_distorted_input.t()
        
        self.H_star *= self.nsamples_star / (self.nsamples_star + tmp)
        self.nsamples_star += tmp
        # inp = inp.float()
        cached_clean_input = math.sqrt(2 / self.nsamples_star) * cached_clean_input.float()
        cached_distorted_input = math.sqrt(2 / self.nsamples_star) * cached_distorted_input.float()
        
        # We are computing H^{*} = (X^{*}X^T - X^{*} X^T)

        # self.H += 2 / self.nsamples_G * inp.matmul(inp.t())
        self.H_star += (cached_clean_input.matmul(cached_distorted_input.t()))
        
        del self.cached_clean_input 
        del self.cached_distorted_input
        self.cached_clean_input = None
        self.cached_distorted_input = None
        
        self.using_G = True
        
    def add_batch_for_upper_bound(self):
        
        assert self.cached_clean_input != None and self.cached_distorted_input != None, "Inputs are not set"
        
        cached_clean_input = self.cached_clean_input
        cached_distorted_input = self.cached_clean_input - self.cached_distorted_input
        
        if len(cached_clean_input.shape) == 2:
            cached_clean_input = cached_clean_input.unsqueeze(0)
        tmp = cached_clean_input.shape[0]
        if len(cached_clean_input.shape) == 3:
            cached_clean_input = cached_clean_input.reshape((-1, cached_clean_input.shape[-1]))
        cached_clean_input = cached_clean_input.t()
        
        if len(cached_distorted_input.shape) == 2:
            cached_distorted_input = cached_distorted_input.unsqueeze(0)
        tmp = cached_distorted_input.shape[0]
        if len(cached_distorted_input.shape) == 3:
            cached_distorted_input = cached_distorted_input.reshape((-1, cached_distorted_input.shape[-1]))
        cached_distorted_input = cached_distorted_input.t()
        
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        cached_clean_input = math.sqrt(2 / self.nsamples) * cached_clean_input.float()
        cached_distorted_input = math.sqrt(2 / self.nsamples) * cached_distorted_input.float()
        
        # We are computing H^{*} = (X^{*}X^T - X^{*} X^T)

        # self.H += 2 / self.nsamples_G * inp.matmul(inp.t())
        self.H += (cached_clean_input.matmul(cached_clean_input.t()) + cached_distorted_input.matmul(cached_distorted_input.t()))
        
        del self.cached_clean_input 
        del self.cached_distorted_input
        self.cached_clean_input = None
        self.cached_distorted_input = None

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        if not getattr(self, "using_G", False):
            return super().fasterquant(
                blocksize=blocksize, percdamp=percdamp, groupsize=groupsize, actorder=actorder, static_groups=static_groups
            )
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        
        # compute the G by G = (- 2 W^{*} H^{*}).t()
        G = (- 2 * W.matmul(self.H_star)).t()
        Hinv_G = - 0.5 * Hinv.matmul(G).t() # should have the same shape as the weight, TODO: check if it is correct to use the transpose
        
        del G # will not be used anymore
        del self.H_star # will not be used anymore

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            dW = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            Hinv_G1 = Hinv_G[:, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.forward(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                
                # check if Err.mal equal to adding dw together
                
                err1 = 0.5 * Hinv_G1[:, i] / d + (w - q) / d
                dw = 0.5 * Hinv_G1[:, i:] + err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                
                W1[:, i:] -= dw # err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= (0.5 * Hinv_G[:, i2:] + Err1.matmul(Hinv[i1:i2, i2:]))

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')


def forward_cache_hessian(
        layer, 
        subset, 
        gptq, 
        inps, 
        outs,
        attention_mask, 
        position_ids, 
        args,
        clean_inps,
        dev,
        scheduler,
        module_input_weighting,
        batch_weighting,
        module_feature_weighting,
        module_sequence_weighting,
        for_validation=False,
        dtype=torch.bfloat16,
    ):
    """
    Forward through the model and cache the Hessian in GPTQ object
    """
    feature_weighting = None
    if module_feature_weighting is not None:
        feature_weighting = layer_wise_get_scores(
            module_feature_weighting, 
            subset, 
            layer, 
            inps, 
            dev, 
            attention_mask, 
            position_ids, 
            args, 
            dtype=dtype,
        )
    
    sequence_weighting = None
    if module_sequence_weighting is not None:
        sequence_weighting = global_get_score(
            module_sequence_weighting, 
            layer, 
            inps, 
            outs, 
            dev,
            method_type="sequence",
        )

    def add_batch(name, for_validation=False):
        def tmp(_, inp, out):
            
            weighting = None
            if args.weighting_apply_module == "all" or any(n in name for n in args.weighting_apply_module.split("|")):
                if scheduler is not None:
                    weighting = scheduler.get_ratio(inp[0].shape[1])
                elif module_input_weighting is not None and args.layerwise_weighting:
                    # weighting = module_input_weighting.compute_weight(layer, inps[j].to(dev), outs[j].to(dev))
                    weighting = module_input_weighting.compute_weight(layer, inp[0].data, out.data)
                elif batch_weighting is not None:
                    weighting = batch_weighting[gptq[name].batch_index]
            if for_validation:
                gptq[name].add_batch_val(
                    inp[0].data, 
                    out.data, 
                    weighting, 
                    feature_weighting[name] if feature_weighting is not None else None,
                    sequence_weighting[gptq[name].batch_index] if sequence_weighting is not None else None,
                )
            else:
                # if args.weighting_apply_module != "all":
                #     print(any(n in name for n in args.weighting_apply_module.split("|")))
                #     if not any(n in name for n in args.weighting_apply_module.split("|")):
                #         import pdb; pdb.set_trace()
                gptq[name].add_batch(
                    inp[0].data, 
                    out.data, 
                    weighting, 
                    feature_weighting[name] if feature_weighting is not None else None,
                    sequence_weighting[gptq[name].batch_index] if sequence_weighting is not None else None,
                )

            gptq[name].batch_index += 1 # using a very hacky way to get batch weighting
        return tmp
    
    handles = []
    for name in subset:
        handles.append(subset[name].register_forward_hook(add_batch(name, for_validation)))

    split = "train" if not for_validation else "val"
    
    # compute the Hessian matrix
    for j in trange(len(inps), desc=f"calc {split} hessian and compute outputs before quantization", leave=False):
        # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layer(inps[j].to(dev, dtype=dtype).unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        
    if args.add_clean_hessian:
        for j in trange(len(inps), desc=f"calc {split} hessian for clean data", leave=False):
            layer(clean_inps[j].to(dev, dtype=dtype).unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            
    if args.add_mixed_hessian:
        for j in trange(len(inps), desc=f"calc {split} hessian for mixed data", leave=False):
            layer((0.5 * clean_inps[j] + 0.5 * inps[j]).to(dev, dtype=dtype).unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

    for h in handles:
        h.remove()
        
    return gptq


def forward_cache_hessian_two_passes(
        layer, 
        subset, 
        gptq, 
        inps, 
        outs,
        attention_mask, 
        position_ids, 
        args,
        clean_inps,
        dev,
        scheduler,
        module_input_weighting,
        batch_weighting,
        **kwargs,
    ):
    """
    Forward through the model and cache the Hessian in GPTQ object
    """
    def cache_distorted_add_batch(name):
        def tmp(_, inp, out):
            gptq[name].set_distorted_input(inp[0].data)
            gptq[name].add_batch(inp[0].data, out.data)
            gptq[name].add_batch_for_G()
            gptq[name].batch_index += 1
        return tmp
    
    def cache_distorted_add_batch_upper_bound(name):
        def tmp(_, inp, out):
            gptq[name].set_distorted_input(inp[0].data)
            gptq[name].add_batch_for_upper_bound()
            gptq[name].batch_index += 1
        return tmp
    
    def cache_clean(name):
        def tmp(_, inp, out):
            gptq[name].set_clean_input(inp[0].data)
        return tmp
    
    if args.clean_corrupted_forward:
        compute_func = cache_distorted_add_batch_upper_bound
    else:
        compute_func = cache_distorted_add_batch

    # compute the Hessian matrix
    for j in trange(len(inps), desc="calc hessian for two passes and compute outputs before quantization", leave=False):
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(cache_clean(name)))

        layer(clean_inps[j].to(dev).unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()
            
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(compute_func(name)))

        layer(inps[j].to(dev).unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()
        
    return gptq


def set_layer(layer, name, target_linear, new_linear):
    
    found_original = False
    for submodule in layer.modules():
        for child_name, child_module in submodule.named_children():
            if child_module is target_linear:
                setattr(submodule, child_name, new_linear)
                found_original = True  # note: do not break to handle tied layers

    assert found_original, f"could not find {name}"


def forward_and_store_outs(layer, inps, outs, dev, attention_mask, position_ids, desc):
    for j in trange(len(inps), desc=desc, leave=False):
        outs_batch = layer(inps[j].to(dev).unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        outs[j].copy_(outs_batch.reshape_as(outs[j]), non_blocking=True)


@torch.no_grad()
def get_inps(
    model: PreTrainedModel,
    data: Sequence,
    model_seqlen: int,
    devices: Sequence[torch.device],
    offload_activations: bool,
) -> Tuple[Sequence[torch.Tensor], Dict]:
    # borrowed and modified from https://github.com/Vahe1994/AQLM/blob/main/main.py
    """
    mocks model launch to collect inputs to the first model layer
    :returns: a list of torch tensors with activations for each device in args.devices.
    Each tensor has shape [nsample_per_device, seq_len, hid_size]
    """
    print("catching layer inputs from data", flush=True)
    layers = get_layers(model)
    device = devices[0] if not offload_activations else torch.device("cpu")

    if isinstance(data, torch.Tensor) and data.shape[0] == 1:  # given a single long tensor, split it into sequences
        assert data.ndim == 2, "data must be either a single tensor with a long sequence or a list of pre-cut sequences"
        num_sequences, num_tokens_dropped = data.numel() // model_seqlen, data.numel() % model_seqlen
        data = [data[:, i * model_seqlen : (i + 1) * model_seqlen].to(device) for i in range(num_sequences)]
        print(f"Got {len(data)} sequences of {model_seqlen} tokens, dropped last {num_tokens_dropped} tokens")
        del num_sequences, num_tokens_dropped

    # data is stored as a list of tuples, [(input, target), ...]
    assert all(sequence[0].shape[1] == model_seqlen for sequence in data)

    emb = model.get_input_embeddings()
    emb_device = emb.weight.device
    if emb_device.type != "cuda":
        emb = emb.to(device)
        # opt has other embeddings
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    device = emb.weight.device  # now default device is the one where the embeddings are.
    layer_device = next(layers[0].parameters()).device
    layers[0] = layers[0].to(device)
    
    if getattr(model.model, 'rotary_emb', None):
        # for llama and qwen models when transformers >=4.45.0
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    dtype = next(iter(model.parameters())).dtype
    nsamples_per_device = (len(data) - 1) // len(devices) + 1
    inps = [
        torch.zeros(
            (min(nsamples_per_device, len(data) - i * nsamples_per_device), model_seqlen, model.config.hidden_size),
            dtype=dtype,
            device=devices[i] if not offload_activations else "cpu",
            pin_memory=offload_activations,
        )
        for i in range(len(devices))
    ]
    forward_arg_names = ["attention_mask", "position_ids"]
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")
        
    cache = {"i": 0, "alibi": None}

    class CatcherExit(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"] // nsamples_per_device][cache["i"] % nsamples_per_device] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise CatcherExit()

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch_inps in data:
        try:
            if isinstance(batch_inps, (list, tuple)):
                batch_inps, *_ = batch_inps
            batch_inps = batch_inps.to(device)
            # call model.forward to trigger the Catcher
            model(batch_inps, attention_mask=torch.ones_like(batch_inps))
        except CatcherExit:
            pass  # exit after catcher finished without running the rest of the model layers

    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_device)
    model.get_input_embeddings().to(emb_device)
    
    if getattr(model.model, 'rotary_emb', None):
        # for llama and qwen models when transformers >=4.45.0
        model.model.rotary_emb = model.model.rotary_emb.to(layer_device)
    
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_device)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_device)
    torch.cuda.empty_cache()
    
    forward_args = {k: cache[k] for k in forward_arg_names}
    assert cache["i"] == sum(len(inp_tensor) for inp_tensor in inps), "internal error: found empty rows in inps"
    return inps, forward_args


def layer_wise_get_scores(module_input_weighting, subset, layer, inps, dev, attention_mask, position_ids, args, dtype=torch.bfloat16):
    
    feature_weighting = {}
    def add_batch(name):
        def tmp(_, inp, out):
            
            weighting = None
            if args.weighting_apply_module == "all" or any(n in name for n in args.weighting_apply_module.split("|")):
                weighting = module_input_weighting.compute_weight(layer, inp[0].data, out.data)
                
            feature_weighting[name].append(weighting)

        return tmp
    
    handles = []
    for name in subset:
        feature_weighting[name] = []
        handles.append(subset[name].register_forward_hook(add_batch(name)))
        
    for j in trange(len(inps), desc=f"compute layer-wise scores", leave=False):
        # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layer(inps[j].to(dev, dtype=dtype).unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        
    for h in handles:
        h.remove()
        
    # take average of the feature weighting
    # module_input_weighting.normalize_weight(torch.stack(feature_weighting[name]).norm(dim=0), module_input_weighting.min_value, module_input_weighting.max_value)
    for name in feature_weighting:
        if feature_weighting[name][0] is not None:
            feature_weighting[name] = module_input_weighting.normalize_weight(
                torch.stack(feature_weighting[name]).mean(dim=0), 
                module_input_weighting.min_value, 
                module_input_weighting.max_value
            )
        else:
            feature_weighting[name] = None
        
    return feature_weighting

def global_get_score(module_input_weighting, layer, inps, outs, dev, method_type="sequence"):
    batch_weighting = []
    for j in range(len(inps)):
        batch_weighting.append(
            module_input_weighting.compute_weight(layer, inps[j].to(dev), outs[j].to(dev))
        )

    if method_type == "sequence":
        batch_weighting = module_input_weighting.normalize_weight(
            torch.hstack(batch_weighting), 
            module_input_weighting.min_value, 
            module_input_weighting.max_value
        )
    else:
        raise ValueError(f"{method_type} is not supported")
    
    return batch_weighting

def get_token_frequency_for_each_data(dataloader) -> list:
    token_freq = defaultdict(int)
    token_freq_per_data = []
    for d in dataloader:
        for token in d[0].flatten():
            token_freq[token.item()] += 1
        
    for d in dataloader:
        token_freq_per_data.append([])
        for token in d[0].flatten():
            token_freq_per_data[-1].append(token_freq[token.item()])
            
            assert token_freq_per_data[-1][-1] != 0, f"token {token.item()} has zero frequency"
            
    return torch.LongTensor(token_freq_per_data)

@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    '''
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    # layers = model.model.layers

    # model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # model.model.norm = model.model.norm.to(dev)
    # layers[0] = layers[0].to(dev)

    # dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros(
    #     (args.nsamples + args.val_size*args.expand_factor, args.train_seqlen, model.config.hidden_size), 
    #     dtype=dtype, device=dev if not args.offload_activations else "cpu",
    #     pin_memory=args.offload_activations
    # )
    # cache = {'i': 0, 'attention_mask': None}

    # class Catcher(nn.Module):
    #     def __init__(self, module):
    #         super().__init__()
    #         self.module = module
    #     def forward(self, inp, **kwargs):
    #         inps[cache['i']] = inp
    #         cache['i'] += 1
    #         cache['attention_mask'] = kwargs['attention_mask']
    #         cache['position_ids'] = kwargs['position_ids']
    #         raise ValueError
    # layers[0] = Catcher(layers[0])
    # for batch in dataloader:
    #     try:
    #         model(batch[0].to(dev))
    #     except ValueError:
    #         pass
    # layers[0] = layers[0].module

    # layers[0] = layers[0].cpu()
    # model.model.embed_tokens = model.model.embed_tokens.cpu()
    # model.model.norm = model.model.norm.cpu()
    
    # torch.cuda.empty_cache()
    
    inps, forward_args = get_inps(
        model,
        dataloader,
        args.train_seqlen,
        devices=[dev],
        offload_activations=args.offload_activations,
    )
    inps = inps[0] # only support one device for now
    layers = get_layers(model)

    token_freq_per_data = get_token_frequency_for_each_data(dataloader)

    outs = torch.zeros_like(inps)
    attention_mask = forward_args['attention_mask'] # should fix this later to be more general for Falcon models
    position_ids = forward_args['position_ids'] # should fix this later to be more general for Falcon models
    
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)
    if position_ids is not None:
        position_ids = position_ids.to(dev)

    quantizers = {}
    sequential = [
                ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
                ['self_attn.o_proj.module'],
                ['mlp.up_proj.module', 'mlp.gate_proj.module'],
                ['mlp.down_proj.module']
            ]
    
    scheduler = None
    if args.scheduler_yaml is not None:
        scheduler = schedulers.load_scheduler(
            args.scheduler_yaml,
            min_value=args.min_value,
            max_value=args.max_value,
            factor=args.factor,
        )
    
    module_input_weighting = None
    batch_weighting = None
    module_feature_weighting = None
    module_sequence_weighting = None
    
    use_clean_pass = args.first_second_order or args.clean_corrupted_forward or args.clean_outs_for_mse or args.clean_outs_for_attn_loss

    indices = torch.randperm(inps.shape[0], device=inps.device)
    inps = inps[indices]
    
    # train_inps, train_outs = inps[args.val_size*args.expand_factor:], outs[args.val_size*args.expand_factor:]
    # val_inps, val_outs = inps[:args.val_size*args.expand_factor], outs[:args.val_size*args.expand_factor]

    if use_clean_pass:
        clean_inps = inps.clone()
        clean_outs = outs.clone()
        
        # clean_train_inps, clean_train_outs = clean_inps[args.val_size*args.expand_factor:], clean_outs[args.val_size*args.expand_factor:]
        # clean_val_inps, clean_val_outs = clean_inps[:args.val_size*args.expand_factor], clean_outs[:args.val_size*args.expand_factor]
    
    for i in range(len(layers)):
        logging.info(f'\nLayer {i}:')
        # print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        original_dtype = next(layer.parameters()).dtype

        org_attn_module = None
        if args.compute_attn_loss:
            org_attn_module = attn_module.CustomLLamaModel(
                copy.deepcopy(layer.input_layernorm), 
                copy.deepcopy(layer.self_attn),
            )
            
        if args.compute_next_attn_loss and i < len(layers) - 1:
            org_attn_module = attn_module.CustomLLamaModel(
                copy.deepcopy(layers[i+1].to(dev).input_layernorm), 
                copy.deepcopy(layers[i+1].to(dev).self_attn),
            )
        
        # clean forward (computed before the layer is quantized)
        # if args.clean_forward == "full":
        # for j in trange(len(clean_inps), desc=f"calc clean outputs before quantization", leave=False):
        #     outs_batch = layer(clean_inps[j].to(dev).unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        #     clean_outs[j].copy_(outs_batch.reshape_as(clean_outs[j]), non_blocking=True)
        
        # already cover both training and validation set
        if use_clean_pass:
            forward_and_store_outs(
                layer, 
                clean_inps, 
                clean_outs, 
                dev, 
                attention_mask, 
                position_ids,
                "calc clean outputs before quantization",
            )
        
        forward_and_store_outs(
            layer, 
            inps, 
            outs, 
            dev, 
            attention_mask, 
            position_ids,
            "calc outputs before quantization",
        )
        # elif args.clean_forward == "half":
        #     for j in trange(len(inps), desc=f"calc {args.clean_forward} clean outputs before quantization", leave=False):
        #         outs_batch = layer(inps[j].to(dev).unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        #         clean_outs[j].copy_(outs_batch.reshape_as(clean_outs[j]), non_blocking=True)

        # if args.pre_compute_act:
        #     print("Pre compute activation")
        #     for j in range(args.nsamples):
        #         outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        
        if (args.module_input_weighting_yaml is not None or args.custom_attn_type is not None) and not args.custom_attn_only_for_cache_output:
            # use cusotm attention to either compute special attentions or force to return attention weights
            attn_module.enable_llama_custom_attention(
                layer, 
                i,
                custom_attn_type=args.custom_attn_type,
                attn_length=args.attn_length,
                num_sink_token=args.num_sink_token,
            )
        
        if args.module_input_weighting_yaml is not None:
            module_input_weighting = input_weighting_module.load_input_weighting_module(
                args.model,
                args.module_input_weighting_yaml,
                method_type=args.adhoc_weighting_method_type,
                half_none_zero_num=args.half_none_zero_num,
                num_bins=args.num_bins,
                min_value=args.min_value,
                max_value=args.max_value,
                masking=args.masking,
                reverse=args.reverse,
                quantile_value=args.quantile_value,
                truncate=args.truncate,
                n_clusters=args.n_clusters,
            )
            
            if not args.layerwise_weighting:
                batch_weighting = []
                for j in range(len(inps)):
                    batch_weighting.append(
                        module_input_weighting.compute_weight(layer, inps[j].to(dev), outs[j].to(dev), token_freq=token_freq_per_data[j].to(dev))
                    )
                    
        if args.feature_weighting_yaml is not None:
            module_feature_weighting = input_weighting_module.load_input_weighting_module(
                args.model,
                args.feature_weighting_yaml,
                max_value=args.feature_max_value,
                reverse=args.feature_reverse,
            )
            
            # normalize later
            module_feature_weighting.normalize = None
            
        if args.sequence_weighting_yaml is not None:
            module_sequence_weighting = input_weighting_module.load_input_weighting_module(
                args.model,
                args.sequence_weighting_yaml,
                max_value=args.sequence_max_value,
                reverse=args.sequence_reverse,
            )
            
            # normalize later
            module_sequence_weighting.normalize = None

        quantized_linears = {}
        
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                if args.wbits_yaml is not None:
                    import yaml
                    layer_weight_bits = yaml.safe_load(open(args.wbits_yaml, "r"))[name]
                else:
                    layer_weight_bits = args.w_bits

                logging.info(f'{name}, bit={layer_weight_bits}')
                # print(f'{name}, bit={layer_weight_bits}', end='  ', flush=True)
                layer_weight_sym = not(args.w_asym)
                
                if i in args.layers_dont_quantize:
                    if args.dont_quantize_qk:
                        if 'q_proj' in name or 'k_proj' in name:
                            layer_weight_bits = 16
                            print(f"Skipping quanitize qk for layer {i}")
                    elif args.dont_quantize_attn:
                        if 'self_attn' in name:
                            layer_weight_bits = 16
                            print(f"Skipping quanitize self_attn for layer {i}")
                    else:
                        layer_weight_bits = 16
                        print(f"Skipping quanitize for layer {i}")

                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                    
                if args.e8p:
                    gptq[name] = ldlq_utils.LDLQ(
                        subset[name], 
                        add_until_fail=args.add_until_fail,
                    )
                    gptq[name].quantizer = ldlq_utils.E8PWeightQuantizer()
                else:
                    if args.first_second_order or args.clean_corrupted_forward:
                        gptq[name] = GPTQ_TP(subset[name])
                    elif args.similarity_term:
                        gptq[name] = GPTQ_sim(subset[name], alpha=args.alpha)
                    elif args.adaptive_gptq and (args.low_rank_before or args.low_rank_after):
                        gptq[name] = AdaptiveLowRankGPTQ(
                            subset[name], 
                            args.normalize_over_tokens,
                            values_to_try=eval(args.values_to_try),
                            train_in_val=args.train_in_val,
                            low_rank_before=args.low_rank_before,
                            low_rank_after=args.low_rank_after,
                            rank_ratio=args.rank_ratio,
                            rank_take_top=args.rank_take_top,
                        )
                    elif args.adaptive_gptq:
                        gptq[name] = AdaptiveGPTQ(
                            subset[name], 
                            args.normalize_over_tokens,
                            values_to_try=eval(args.values_to_try),
                            train_in_val=args.train_in_val,
                        )
                    else:
                        gptq[name] = GPTQ(
                            subset[name], 
                            args.normalize_over_tokens, 
                            args.normalize_hessian,
                            add_until_fail=args.add_until_fail,
                            low_rank_before=args.low_rank_before,
                            low_rank_after=args.low_rank_after,
                            rank_ratio=args.rank_ratio,
                            rank_take_top=args.rank_take_top,
                        )

                    gptq[name].quantizer = quant_utils.WeightQuantizer()

                gptq[name].quantizer.configure(
                    layer_weight_bits, 
                    perchannel=True, 
                    sym=layer_weight_sym, 
                    mse=args.w_clip, 
                    scale_override=args.e8p_scale_override,
                )

                gptq[name].batch_index = 0 # using a very hacky way to get batch weighting
                
            if args.first_second_order or args.clean_corrupted_forward:
                cache_hessian_function = forward_cache_hessian_two_passes
            else:
                cache_hessian_function = forward_cache_hessian

            # compute train Hessian
            gptq = cache_hessian_function(
                layer, 
                subset, 
                gptq, 
                inps[args.val_size*args.expand_factor:], 
                outs[args.val_size*args.expand_factor:],
                attention_mask, 
                position_ids, 
                args,
                clean_inps[args.val_size*args.expand_factor:] if use_clean_pass else None,
                dev,
                scheduler,
                module_input_weighting,
                batch_weighting[args.val_size*args.expand_factor:] if batch_weighting else None,
                module_feature_weighting,
                module_sequence_weighting,
                dtype=original_dtype,
            )
            
            if args.adaptive_gptq:
                # compute val Hessian
                gptq = cache_hessian_function(
                    layer, 
                    subset, 
                    gptq, 
                    inps[:args.val_size*args.expand_factor], 
                    outs[:args.val_size*args.expand_factor],
                    attention_mask, 
                    position_ids, 
                    args,
                    clean_inps[:args.val_size*args.expand_factor] if use_clean_pass else None,
                    dev,
                    scheduler,
                    module_input_weighting,
                    batch_weighting[:args.val_size*args.expand_factor] if batch_weighting else None,
                    module_feature_weighting,
                    for_validation=True,
                )

            for name in subset:
                # # use this to make sure all samples are processed for every module
                # assert gptq[name].batch_index == args.nsamples
                
                if args.adhoc_multiplier:
                    if name == "mlp.down_proj.module" and i == 1:
                        torch.save(gptq[name].H, f"/nas-hdd/ylsung/gptq_hessian/mlp_down_seed{args.seed}.pth")
                        exit()
                        multiplier = 21
                    else:
                        multiplier = 1
                    
                    print(multiplier)
                else:
                    multiplier = 1
                
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp * multiplier, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False,
                    quant=not args.no_gptq,
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                
                # print(gptq[name].layer.weight.abs().mean().item())
                
                # add quantized linear for fine-tuning
                quantized_linears[name] = gptq[name].get_quantize_linear(qat=args.qat)
                
                gptq[name].free()
                
        if (args.module_input_weighting_yaml is not None or args.custom_attn_type is not None) and args.custom_attn_only_for_cache_output:
            attn_module.enable_llama_custom_attention(
                layer, 
                i,
                custom_attn_type=args.custom_attn_type,
                attn_length=args.attn_length,
            )
                
        # if args.custom_attn_type is not None and not args.custom_attn_for_cache_output:
        #     # the output computed using standard attention
        #     attn_module.disable_llama_custom_attention(
        #         layer, 
        #     )
        
        # change the standard layer to quantized layer
        for names in sequential:
            subset = {n: full[n] for n in names}
            for name in subset:
                set_layer(layer, name, subset[name], quantized_linears[name])
        
        del gptq
        torch.cuda.empty_cache()

        if args.optimizer_yaml is not None:
            optimizer = optimizers.load_optimizer(
                args.optimizer_yaml, quant_lr=args.quant_lr, attn_loss_ratio=args.attn_loss_ratio
                )
            
            quant_attn_module = None
            if args.compute_attn_loss:
                # create a new attn module so that the forward function can be modified
                # but the weights are link to the quantized weights
                self_attn = copy.deepcopy(layer.self_attn)
                # link their weights
                self_attn.q_proj = layer.self_attn.q_proj
                self_attn.k_proj = layer.self_attn.k_proj
                self_attn.v_proj = layer.self_attn.v_proj
                self_attn.o_proj = layer.self_attn.o_proj

                quant_attn_module = attn_module.CustomLLamaModel(
                    layer.input_layernorm, 
                    self_attn,
                )

            layer = optimizer.finetune(
                layer,
                dev,
                args,
                train_inps=inps[args.val_size*args.expand_factor:],
                train_outs=outs[args.val_size*args.expand_factor:],
                train_clean_inps=clean_inps[args.val_size*args.expand_factor:] if use_clean_pass else None,
                train_clean_outs=clean_outs[args.val_size*args.expand_factor:] if use_clean_pass else None,
                val_inps=inps[:args.val_size*args.expand_factor],
                val_outs=outs[:args.val_size*args.expand_factor],
                val_clean_inps=clean_inps[:args.val_size*args.expand_factor] if use_clean_pass else None,
                val_clean_outs=clean_outs[:args.val_size*args.expand_factor] if use_clean_pass else None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                org_attn_module=org_attn_module,
                quant_attn_module=quant_attn_module,
            )

        # print(layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0])
        # change back to standard layer
        for names in sequential:
            subset = {n: full[n] for n in names}
            for name in subset:
                set_layer(layer, name, quantized_linears[name], quantized_linears[name].to_fake_quant_linear())
        
        # IMPORTANT!!!
        # link the ActQuantWrapper.weight to ActQuantWrapper.module.weight
        # as well as bias, or it will cause twice the storage
        for _, wrapper in quant_utils.find_qlayers(layer, layers=[quant_utils.ActQuantWrapper]).items():
            wrapper.weight = wrapper.module.weight
            wrapper.bias = wrapper.module.bias
            
        for _, wrapper in quant_utils.find_qlayers(layer, layers=[quant_utils.ActQuantWrapper]).items():
            assert wrapper.weight is wrapper.module.weight
            assert wrapper.bias is wrapper.module.bias

        if not args.pre_compute_act:
            forward_and_store_outs(
                layer, 
                inps, 
                outs, 
                dev,
                attention_mask, 
                position_ids,
                "calc outs after quantization",
            )
        
        if args.module_input_weighting_yaml is not None or args.custom_attn_type is not None:
            # the output computed using custom attention
            attn_module.disable_llama_custom_attention(
                layer, 
            )

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        
        # print(outs[0])
        # print(clean_outs[0])
        # print((outs[0] - clean_outs[0]).norm(dim=-1))
        
        inps, outs = outs, inps
        
        if use_clean_pass:
            clean_inps, clean_outs = clean_outs, clean_inps
        
    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers

       
@torch.no_grad()
def rtn_fwrd(model, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(layer,
                                            layers=[torch.nn.Linear])

        for name in subset:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and 'down_proj' in name:
                layer_weight_bits = 8

            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.forward(W).to(
                next(iter(layer.parameters())).dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
            
    utils.cleanup_memory(verbos=True)
    return quantizers

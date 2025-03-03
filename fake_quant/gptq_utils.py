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
        add_until_fail=False,
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.add_until_fail = add_until_fail

    def add_batch(self, inp, out, weighting=None):
        
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

        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

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
        else:
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
        
    def get_quantize_linear(self, qat=False):
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


def forward_cache_hessian(
        layer, 
        subset, 
        gptq, 
        inps, 
        outs,
        attention_mask, 
        position_ids, 
        args,
        dev,
        batch_weighting,
        dtype=torch.bfloat16,
    ):
    """
    Forward through the model and cache the Hessian in GPTQ object
    """
    def add_batch(name):
        def tmp(_, inp, out):
            
            weighting = None
            if args.weighting_apply_module == "all" or any(n in name for n in args.weighting_apply_module.split("|")):
                if batch_weighting is not None:
                    weighting = batch_weighting[gptq[name].batch_index]

            gptq[name].add_batch(
                inp[0].data, 
                out.data, 
                weighting, 
            )

            gptq[name].batch_index += 1 # using a very hacky way to get batch weighting
        return tmp
    
    handles = []
    for name in subset:
        handles.append(subset[name].register_forward_hook(add_batch(name)))

    split = "train"
    
    # compute the Hessian matrix
    for j in trange(len(inps), desc=f"calc {split} hessian and compute outputs before quantization", leave=False):
        # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layer(inps[j].to(dev, dtype=dtype).unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

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
    
    module_input_weighting = None
    batch_weighting = None
    
    indices = torch.randperm(inps.shape[0], device=inps.device)
    inps = inps[indices]

    for i in range(len(layers)):
        logging.info(f'\nLayer {i}:')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        original_dtype = next(layer.parameters()).dtype
        
        forward_and_store_outs(
            layer, 
            inps, 
            outs, 
            dev, 
            attention_mask, 
            position_ids,
            "calc outputs before quantization",
        )

        if args.module_input_weighting_yaml:
            # use cusotm attention to either compute special attentions or force to return attention weights
            attn_module.enable_llama_custom_attention(
                layer, 
                i,
                custom_attn_type=args.custom_attn_type,
                attn_length=args.attn_length,
                num_sink_token=args.num_sink_token,
            )
        
            module_input_weighting = input_weighting_module.load_input_weighting_module(
                args.model,
                args.module_input_weighting_yaml,
                method_type=args.adhoc_weighting_method_type,
                num_bins=args.num_bins,
                min_value=args.min_value,
                max_value=args.max_value,
                masking=args.masking,
                reverse=args.reverse,
                quantile_value=args.quantile_value,
                truncate=args.truncate,
            )
            
            batch_weighting = []
            for j in range(len(inps)):
                batch_weighting.append(
                    module_input_weighting.compute_weight(layer, inps[j].to(dev), outs[j].to(dev), token_freq=token_freq_per_data[j].to(dev))
                )

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
                    gptq[name] = GPTQ(
                        subset[name], 
                        add_until_fail=args.add_until_fail,
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
                
            # compute train Hessian
            gptq = forward_cache_hessian(
                layer, 
                subset, 
                gptq, 
                inps, 
                outs,
                attention_mask, 
                position_ids, 
                args,
                dev,
                batch_weighting if batch_weighting else None,
                dtype=original_dtype,
            )

            for name in subset:
                # # use this to make sure all samples are processed for every module
                # assert gptq[name].batch_index == args.nsamples

                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                
                # print(gptq[name].layer.weight.abs().mean().item())
                
                # add quantized linear for fine-tuning
                quantized_linears[name] = gptq[name].get_quantize_linear()
                
                gptq[name].free()

        # change the standard layer to quantized layer
        for names in sequential:
            subset = {n: full[n] for n in names}
            for name in subset:
                set_layer(layer, name, subset[name], quantized_linears[name])
        
        del gptq
        torch.cuda.empty_cache()

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

        forward_and_store_outs(
            layer, 
            inps, 
            outs, 
            dev,
            attention_mask, 
            position_ids,
            "calc outs after quantization",
        )
    
        if args.module_input_weighting_yaml:
            # the output computed using custom attention
            attn_module.disable_llama_custom_attention(
                layer, 
            )

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        
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

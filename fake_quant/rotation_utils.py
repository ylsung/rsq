import model_utils
import torch
import typing
import utils
import transformers
import tqdm, math
import quant_utils
from hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2
from fast_hadamard_transform import hadamard_transform

from model_utils import (
    get_layers,
)
import logging

from gptq_utils import get_inps, forward_and_store_outs, forward_cache_hessian
from gptq_modules import GPTQ_bf16


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear], cast_back=True) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        if cast_back:
            linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
        else:
            linear.weight.data = (W_ / layernorm.weight.double()).to(torch.float32)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            if cast_back:
                linear.bias.data = linear.bias.data.to(linear_dtype)
            else:
                linear.bias.data = linear.bias.data.to(torch.float32)
            
def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model):
    
    model_type = model_utils.get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion
    for W in model_utils.get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    layers = model_utils.get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # i_e = torch.randn(1, 10, 3584).bfloat16()
        # pre_out = layer(i_e, position_ids=torch.arange(10).unsqueeze(0).long())[0]
        
        
        # def compute_diffs():
        #     return (
        #         layer.self_attn.q_proj(layer.input_layernorm(i_e)), 
        #         layer.self_attn.k_proj(layer.input_layernorm(i_e)), 
        #         layer.self_attn.v_proj(layer.input_layernorm(i_e)),
        #         layer.mlp.up_proj(layer.post_attention_layernorm(i_e)),
        #         layer.mlp.gate_proj(layer.post_attention_layernorm(i_e)),
        #     )
        
        # outs = compute_diffs()

        # fuse the input layernorms into the linear layers
        if any(model_type == n for n in [model_utils.LLAMA_MODEL, model_utils.QWEN2_MODEL, model_utils.MISTRAL_MODEL]):
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')

        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
            
        
        # if model_type == model_utils.LLAMA_MODEL:
        #     rms_norm_class = transformers.models.llama.modeling_llama.LlamaRMSNorm
        # elif model_type == model_utils.QWEN2_MODEL:
        #     rms_norm_class = transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
        # else:
        #     rms_norm_class = torch.nn.LayerNorm
        
        # model_utils.replace_modules(
        #     layer,
        #     rms_norm_class,
        #     lambda _: model_utils.RMSN(model.config.hidden_size, eps=getattr(model.config, "rms_norm_eps", 1e-5)),
        #     replace_layers=False,
        # )
        
        # post_out = layer(i_e, position_ids=torch.arange(10).unsqueeze(0).long())[0]
        
        # value = (pre_out - post_out).abs().max()
        
        # # print("layer_diff:", value)
        
        # post_outs = compute_diffs()
        
        # print("attn_diff:")
        
        # for o1, o2 in zip(outs, post_outs):
        #     print((o1 - o2).abs().max())
        
        # if value > 0.7:
        #     import pdb; pdb.set_trace()
        
    fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
    
    if model_type == model_utils.LLAMA_MODEL:
        rms_norm_class = transformers.models.llama.modeling_llama.LlamaRMSNorm
    elif model_type == model_utils.QWEN2_MODEL:
        rms_norm_class = transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
    elif model_type == model_utils.MISTRAL_MODEL:
        rms_norm_class = transformers.models.mistral.modeling_mistral.MistralRMSNorm
    else:
        rms_norm_class = torch.nn.LayerNorm
    
    model_utils.replace_modules(
        model,
        rms_norm_class,
        lambda _: model_utils.RMSN(model.config.hidden_size, eps=getattr(model.config, "rms_norm_eps", 1e-5)),
        replace_layers=False,
    )
    

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_orthogonal_matrix(size, mode, device=utils.DEV):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')

    

def rotate_embeddings(model, Q: torch.Tensor, cast_back=True) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        if cast_back:
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        else:
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=torch.float32)

    
def rotate_attention_inputs(layer, Q, model_type, cast_back=True) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        if cast_back:
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        else:
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=torch.float32)
    
        # only for casting dtype
        if W.bias is not None:
            if cast_back:
                W.bias.data = W.bias.data.to(device="cpu", dtype=dtype)
            else:
                W.bias.data = W.bias.data.to(device="cpu", dtype=torch.float32)
        
def rotate_attention_output(layer, Q, model_type, cast_back=True) -> None:
    # Rotate output matrix of the self-attention layer.
    if any(model_type == n for n in [model_utils.LLAMA_MODEL, model_utils.QWEN2_MODEL, model_utils.MISTRAL_MODEL]):
        W = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    if cast_back:
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    else:
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=torch.float32)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        if cast_back:
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
        else:
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=torch.float32)

def rotate_mlp_input(layer, Q, model_type, cast_back=True):
    # Rotate the MLP input weights.
    if any(model_type == n for n in [model_utils.LLAMA_MODEL, model_utils.QWEN2_MODEL, model_utils.MISTRAL_MODEL]):
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        if cast_back:
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        else:
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=torch.float32)
        
        # only for casting dtype
        if W.bias is not None:
            if cast_back:
                W.bias.data = W.bias.data.to(device="cpu", dtype=dtype)
            else:
                W.bias.data = W.bias.data.to(device="cpu", dtype=torch.float32)
    
def rotate_mlp_output(layer, Q, model_type, cast_back=True):
    # Rotate the MLP output weights and bias.
    if any(model_type == n for n in [model_utils.LLAMA_MODEL, model_utils.QWEN2_MODEL, model_utils.MISTRAL_MODEL]):
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    if cast_back:
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    else:
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=torch.float32)
    
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        if cast_back:
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
        else:
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=torch.float32)
            
    apply_exact_had_to_linear(W, had_dim=-1, output=False, cast_back=cast_back) #apply exact (inverse) hadamard on the weights of mlp output


def apply_exact_had_to_linear_mlp_output(layer, model_type, cast_back=True):
    # Rotate the MLP output weights and bias.
    if any(model_type == n for n in [model_utils.LLAMA_MODEL, model_utils.QWEN2_MODEL, model_utils.MISTRAL_MODEL]):
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    apply_exact_had_to_linear(W, had_dim=-1, output=False, cast_back=cast_back) #apply exact (inverse) hadamard on the weights of mlp output

def matmul_hadU_cuda_had(X, hadK, transpose=False):
    '''
    Apply hadamard transformation. 
    It reshapes X and applies Walsh-Hadamard transform to the last dimension. 
    Then, it will multiply the retult by another hadamard matrix.
    '''
    from fast_hadamard_transform import hadamard_transform
    from hadamard_utils import get_had172
    n = X.shape[-1]
    K = hadK.shape[-1]

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().cuda().view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), scale=1/math.sqrt(n))
    input = hadK.to(input.device).to(input.dtype) @ input 
    return input.to(X.device).to(X.dtype).reshape(
        X.shape) 

def rotate_faster_down_proj(layer, model_type, hardK):
    from fast_hadamard_transform import hadamard_transform
    if any(model_type == n for n in [model_utils.LLAMA_MODEL, model_utils.QWEN2_MODEL, model_utils.MISTRAL_MODEL]):
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Faster MLP is onlu supported for LLaMa models!')
    
    dtype = W.weight.data.dtype
    W.weight.data = matmul_hadU_cuda_had(W.weight.data.float().cuda(), hardK)
    W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)


def rotate_head(model, Q: torch.Tensor, cast_back=True) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    if cast_back:
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    else:
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=torch.float32)


def rotate_ov_proj(layer, model_type, head_num, head_dim, cast_back=True):
    v_proj = layer.self_attn.v_proj
    if any(model_type == n for n in [model_utils.LLAMA_MODEL, model_utils.QWEN2_MODEL, model_utils.MISTRAL_MODEL]):
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        o_proj = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, cast_back=cast_back)
    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False, cast_back=cast_back)


@torch.inference_mode()
def rotate_model(model, args):
    Q = get_orthogonal_matrix(model.config.hidden_size,
                                                args.rotate_mode)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    
    if model_type == model_utils.MISTRAL_MODEL:
        # TODO: check if need to change to for all models
        head_dim = config.head_dim

    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type)
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)


@torch.inference_mode()
def post_process_model_after_load(model, args):
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    model_type = model_utils.model_type_extractor(model)
    
    if model_type == model_utils.MISTRAL_MODEL:
        head_dim = config.head_dim

    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        apply_exact_had_to_linear_mlp_output(layers[idx], model_type)
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)

@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]

def register_online_rotation(module, Q:torch.Tensor):
    assert not hasattr(module, 'Q')
    module.register_buffer('Q', Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)


class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs['k_groupsize'] in [-1, head_dim], f'Only token-wise/{head_dim}g quantization is supported for K-cache'
            self.k_bits = kwargs['k_bits']
            self.k_groupsize = kwargs['k_groupsize']
            self.k_sym = kwargs['k_sym']
            self.k_clip_ratio = kwargs['k_clip_ratio']
            self.k_quantizer.configure(bits=self.k_bits, groupsize=-1, #we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                                   sym=self.k_sym, clip_ratio=self.k_clip_ratio)

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = hadamard_transform(q.float(), scale=1/math.sqrt(q.shape[-1])).to(dtype)
        k = hadamard_transform(k.float(), scale=1/math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape
        

        if self.k_groupsize == -1: #token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, self.config.hidden_size)
            self.k_quantizer.find_params(token_wise_k)
            k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        else: #head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)
        
        self.k_quantizer.free()
            
        return q, k



def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward. 
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    import monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name, functools.partial(QKRotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)
    
    
@torch.no_grad()
def rotate_model_gptq(model, dataloader, dev, args):
    logging.info('-----GPTQ Rotation-----')
    
    Q = get_orthogonal_matrix(model.config.hidden_size,
                                                args.rotate_mode)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    model_type = model_utils.model_type_extractor(model)
    
    if model_type == model_utils.MISTRAL_MODEL:
        # TODO: check if need to change to for all models
        head_dim = config.head_dim
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    model_type = model_utils.get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion
    for W in model_utils.get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
        
    rotate_embeddings(model, Q)
    rotate_head(model, Q)

    inps, forward_args = get_inps(
        model,
        dataloader,
        args.train_seqlen,
        devices=[dev],
        offload_activations=args.offload_activations,
    )
    inps = inps[0] # only support one device for now
    layers = get_layers(model)

    outs = torch.zeros_like(inps)
    attention_mask = forward_args['attention_mask'] # should fix this later to be more general for Falcon models
    position_ids = forward_args['position_ids'] # should fix this later to be more general for Falcon models
    
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)
    if position_ids is not None:
        position_ids = position_ids.to(dev)

    quantizers = {}
    # remove the .module because we are using the original model directly
    sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]

    indices = torch.randperm(inps.shape[0], device=inps.device)
    inps = inps[indices]
    
    for i in range(len(layers)):
        logging.info(f'\nLayer {i}:')
        # print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i]
        original_dtype = next(layer.parameters()).dtype
        # fuse the input layernorms into the linear layers
        if any(model_type == n for n in [model_utils.LLAMA_MODEL, model_utils.QWEN2_MODEL, model_utils.MISTRAL_MODEL]):
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj], cast_back=False)    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj], cast_back=False)
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj], cast_back=False)
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1], cast_back=False)
        else:
            raise ValueError(f'Unknown model type {model_type}')
        
        rotate_attention_inputs(layer, Q, model_type, cast_back=False)
        rotate_attention_output(layer, Q, model_type, cast_back=False)
        rotate_mlp_input(layer, Q, model_type, cast_back=False)
        rotate_mlp_output(layer, Q, model_type, cast_back=False)
        rotate_ov_proj(layer, model_type, num_heads, head_dim, cast_back=False)
        
        layer = layer.to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ_bf16(
                    subset[name], 
                )

            # compute train Hessian
            gptq = forward_cache_hessian(
                layer, 
                subset, 
                gptq, 
                inps[args.val_size:], 
                outs[args.val_size:],
                attention_mask, 
                position_ids, 
                args,
                None,
                dev,
                None,
                None,
                None,
                dtype=torch.float32,
            )
            
            for name in subset:
                # # use this to make sure all samples are processed for every module
                # assert gptq[name].batch_index == args.nsamples
                
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False,
                    quant=not args.no_gptq, original_dtype=original_dtype,
                )
                gptq[name].free()
        
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                subset[name].weight.data = subset[name].weight.data.to(original_dtype)
                if hasattr(subset[name], 'bias') and subset[name].bias is not None:
                    subset[name].bias.data = subset[name].bias.data.to(original_dtype)
                
        del gptq
        torch.cuda.empty_cache()

        forward_and_store_outs(
            layer, 
            inps, 
            outs, 
            dev,
            attention_mask, 
            position_ids,
            "calc outs after quantization",
        )

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        
    if model_type == model_utils.LLAMA_MODEL:
        rms_norm_class = transformers.models.llama.modeling_llama.LlamaRMSNorm
    elif model_type == model_utils.QWEN2_MODEL:
        rms_norm_class = transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
    elif model_type == model_utils.MISTRAL_MODEL:
        rms_norm_class = transformers.models.mistral.modeling_mistral.MistralRMSNorm
    else:
        rms_norm_class = torch.nn.LayerNorm
    
    model_utils.replace_modules(
        model,
        rms_norm_class,
        lambda _: model_utils.RMSN(model.config.hidden_size, eps=getattr(model.config, "rms_norm_eps", 1e-5)),
        replace_layers=False,
    )
    
    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Rotation Done-----\n')
    return quantizers

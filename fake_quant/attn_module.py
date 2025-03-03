import math
import types
import warnings
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)   

from transformers.utils import (
    logging,
)

from transformers.cache_utils import Cache

from transformers.cache_utils import StaticCache
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_with_cache_position


class CustomLLamaModel(nn.Module):
    # For transformers=4.40.0
    def __init__(self, input_layernorm, self_attn):
        super().__init__()
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.self_attn.forward = types.MethodType(llama_extract_attn_weights, self.self_attn)
        
    def _get_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_seen_tokens: int,
    ):
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice
                
        return causal_mask

    def forward(self, inputs_embeds, attention_mask=None, position_ids=None):
        output_attentions = False
        cache_position = None
        
        past_seen_tokens = 0

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._get_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)
    
        hidden_states = self.input_layernorm(inputs_embeds)
        
        self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=output_attentions,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=None,
        )
        
        return self_attn_weights


def llama_extract_attn_weights(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    past_key_value = getattr(self, "past_key_value", past_key_value)
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
        
    return attn_weights



def convert_to_block_attn(attn, n, min_dtype):
    T = attn.size(-2)
    device = attn.device

    # Create an index tensor from 0 to T-1
    indices = torch.arange(T, device=device)

    # Compute block numbers for each token
    block_i = (indices.unsqueeze(1) // n)  # Shape: [T, 1]
    block_j = (indices.unsqueeze(0) // n)  # Shape: [1, T]

    # Create a attn where:
    # - Tokens are in the same block (block_i == block_j)
    # - j <= i (causal within the block)
    allowed = (block_i == block_j) & (indices.unsqueeze(1) >= indices.unsqueeze(0))
    
    # Apply the allowed attn:
    # - Set allowed positions to 0.0
    # - Set disallowed positions to -inf
    attn.masked_fill_(~allowed, min_dtype)
    

def convert_to_window_attn(attn, n, min_dtype):
    T = attn.size(-2)
    device = attn.device

    # Create an index tensor from 0 to T-1
    indices = torch.arange(T, device=device)

    # Expand indices to create row and column indices
    row_idx = indices.unsqueeze(1)  # Shape: [T, 1]
    col_idx = indices.unsqueeze(0)  # Shape: [1, T]

    # Calculate the distance between query and key tokens
    distance = row_idx - col_idx  # Shape: [T, T]
    
    # Define allowed positions: within the window and causal
    allowed = (distance < n) & (distance >= 0)

    # Apply the allowed attn:
    # - Set allowed positions to 0.0
    # - Set disallowed positions to -inf
    attn.masked_fill_(~allowed, min_dtype)
    

def convert_to_topk_attn(attn, n, min_dtype):
    T = attn.size(-2)
    device = attn.device
    indices = torch.arange(T, device=device)
    # Step 3: Mask value_map to ignore j > i by setting them to -inf
    # This ensures that topk only considers j < i
    # Step 4: For each row, find the top_n indices based on adjusted_value_map
    # Handle cases where top_n > available tokens by setting k=min(top_n, i)
    # torch.topk will handle k > available tokens by padding with -inf
    topk_scores, topk_indices = torch.topk(attn, k=n, dim=-1, largest=True, sorted=False)

    # Step 5: Create a mask where top_n positions are allowed
    allowed = torch.zeros_like(attn, dtype=torch.bool)  # Initialize all to False
    allowed.scatter_(-1, topk_indices, True)  # Set top_n positions to True
    
    # Step 6: Mask always allow attending to self
    if len(allowed.shape) == 2:
        allowed[indices, indices] = True
        assert (torch.diagonal(allowed, offset=0, dim1=0, dim2=1) == 1).all()
    elif len(allowed.shape) == 3:
        allowed[:, indices, indices] = True
        assert (torch.diagonal(allowed, offset=0, dim1=1, dim2=2) == 1).all()
    elif len(allowed.shape) == 4:
        allowed[:, :, indices, indices] = True
        assert (torch.diagonal(allowed, offset=0, dim1=2, dim2=3) == 1).all()
    
    attn.masked_fill_(~allowed, min_dtype)
    
    
def convert_to_sink_attn(attn, n, n_sink_tokens, min_dtype):
    T = attn.size(-2)
    device = attn.device

    # Create an index tensor from 0 to T-1
    indices = torch.arange(T, device=device)

    # Expand indices to create row and column indices
    row_idx = indices.unsqueeze(1)  # Shape: [T, 1]
    col_idx = indices.unsqueeze(0)  # Shape: [1, T]

    # Calculate the distance between query and key tokens
    distance = row_idx - col_idx  # Shape: [T, T]
    
    # Define allowed positions: within the window and causal
    allowed = (distance < n - n_sink_tokens) & (distance >= 0)
    allowed[:, :n_sink_tokens] = True
    allowed = allowed & (distance >= 0)
    
    # Apply the allowed attn:
    # - Set allowed positions to 0.0
    # - Set disallowed positions to -inf
    attn.masked_fill_(~allowed, min_dtype)
    

def convert_to_shift_attn(attn, n, min_dtype):
    T = attn.size(-2)
    device = attn.device

    # Create an index tensor from 0 to T-1
    indices = torch.arange(T, device=device)
    
    assert n % 2 == 0
    
    indices = indices.roll(n//2)
    # indices = torch.cat([indices[-n//2:], indices[:-n//2]], dim=0)
    # indices = torch.cat([indices[n//2:], indices[:n//2]], dim=0)

    # Compute block numbers for each token
    block_i = (indices.unsqueeze(1) // n)  # Shape: [T, 1]
    block_j = (indices.unsqueeze(0) // n)  # Shape: [1, T]
    
    # Create a attn where:
    # - Tokens are in the same block (block_i == block_j)
    # - j <= i (causal within the block)
    
    mask = indices.unsqueeze(1) >= indices.unsqueeze(0)

    # Reorder the rows and columns
    # new_order = torch.cat([torch.arange(n//2, T, device=device), torch.arange(n//2, device=device)])
    new_order = indices.roll(-n)
    mask = mask[new_order][:, new_order]

    allowed = (block_i == block_j) & mask
    
    # Apply the allowed attn:
    # - Set allowed positions to 0.0
    # - Set disallowed positions to -inf
    attn.masked_fill_(~allowed, min_dtype)
    

def get_causal_mask_4_45(
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    if using_static_cache:
        target_length = past_key_values.get_max_length()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        device=device,
        min_dtype=min_dtype,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
    )
    
    return causal_mask


def llama_custom_attention_forward_4_45(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()

    # if self.config.pretraining_tp > 1:
    #     key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
    #     query_slices = self.q_proj.weight.split(
    #         (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
    #     )
    #     key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
    #     value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

    #     query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
    #     query_states = torch.cat(query_states, dim=-1)

    #     key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
    #     key_states = torch.cat(key_states, dim=-1)

    #     value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
    #     value_states = torch.cat(value_states, dim=-1)

    # else:
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        # logger.warning_once(
        #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
        #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
        #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
        #     "removed and `position_embeddings` will be mandatory."
        # )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    
    if attention_mask is None:
        # create causal mask ourself, because Sdpa/Flash attention disables it
        if cache_position is None:
            past_seen_tokens = past_key_value.get_seq_length() if past_key_value is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        attention_mask = get_causal_mask_4_45(
            attention_mask,
            hidden_states,
            cache_position,
            past_key_value,
            False,
        )

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    min_dtype = torch.finfo(attn_weights.dtype).min

    # in-place operation
    if self.custom_attn_type == "block":
        convert_to_block_attn(attn_weights, self.attn_length, min_dtype)
    elif self.custom_attn_type == "window":
        convert_to_window_attn(attn_weights, self.attn_length, min_dtype)
    elif self.custom_attn_type == "topk":
        convert_to_topk_attn(attn_weights, self.attn_length, min_dtype)
    elif self.custom_attn_type == "sink":
        convert_to_sink_attn(attn_weights, self.attn_length, self.num_sink_token, min_dtype)
    elif self.custom_attn_type == "ss":
        num_heads = attn_weights.size(1)
        convert_to_block_attn(attn_weights[:, :num_heads//2], self.attn_length, min_dtype)
        convert_to_shift_attn(attn_weights[:, num_heads//2:], self.attn_length, min_dtype)
        
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    # if self.config.pretraining_tp > 1:
    #     attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
    #     o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
    #     attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    # else:
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def enable_llama_custom_attention(
    layer, 
    layer_id,
    custom_attn_type=None,
    attn_length=None,
    num_sink_token=8,
):
    """
    replace the forward function of LlamaAttention with a custom forward function `llama_custom_attention_forward`
    """
    modified_module = layer.self_attn
    modified_module.layer_id = layer_id
    
    # None don't modify the attention but only for input_weighting_module to get the attention weight
    # because default attn won't return attention weights
    assert custom_attn_type in [None, "block", "window", "topk", "sink", "ss"]
    
    if custom_attn_type is not None:
        assert attn_length is not None

    modified_module.custom_attn_type = custom_attn_type
    modified_module.attn_length = attn_length
    modified_module.num_sink_token = num_sink_token
    
    modified_module.original_forward = modified_module.forward
    modified_module.forward = types.MethodType(llama_custom_attention_forward_4_45, modified_module)
    
    return modified_module


def disable_llama_custom_attention(
    layer, 
):
    modified_module = layer.self_attn
    modified_module.forward = modified_module.original_forward
    
    del modified_module.custom_attn_type
    del modified_module.attn_length
    del modified_module.num_sink_token
    del modified_module.original_forward

    return modified_module
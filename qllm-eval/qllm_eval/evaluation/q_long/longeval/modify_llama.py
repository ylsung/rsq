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
    rotate_half,
)   

from transformers.utils import (
    logging,
)
import copy


logger = logging.get_logger(__name__)


def llama_custom_flash_attention_forward_4_40_extraction(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    # hooked query_states and key_states to recover the perfect attention matrices
    # self.query_states_cache = query_states.detach()
    # self.key_states_cache = key_states.detach()
    self.hidden_states_cache = hidden_states.detach()

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_custom_flash_attention_forward_4_40_apply(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    # load the hooked query_states and key_states to recover the perfect attention matrices
    
    guide_hidden_states = copy.deepcopy(self.guide_layer.hidden_states_cache)
    
    assert 0 <= self.top_k <= 1
    
    if self.magnitude_dim == "token":
        magnitude = guide_hidden_states.norm(dim=-1)
        num_set_to_distorted = int(magnitude.shape[-1] * (1 - self.top_k))
        indices_set_to_distorted = torch.topk(magnitude, num_set_to_distorted, largest=False).indices
        guide_hidden_states[:, indices_set_to_distorted[-1], :] = hidden_states[:, indices_set_to_distorted[-1], :]
    elif self.magnitude_dim == "channel":
        magnitude = guide_hidden_states.norm(dim=1)
        num_set_to_distorted = int(magnitude.shape[1] * (1 - self.top_k))
        indices_set_to_distorted = torch.topk(magnitude, num_set_to_distorted, largest=False).indices
        guide_hidden_states[:, :, indices_set_to_distorted[-1]] = hidden_states[:, :, indices_set_to_distorted[-1]]

    if "q" in self.apply_module:
        q_guide_layer = self.q_proj if self.using_distorted_guide_layer else self.guide_layer.q_proj
        query_states = q_guide_layer(guide_hidden_states)
    else:
        query_states = self.q_proj(hidden_states)
    
    if "k" in self.apply_module:
        k_guide_layer = self.k_proj if self.using_distorted_guide_layer else self.guide_layer.k_proj
        key_states = k_guide_layer(guide_hidden_states)
    else:
        key_states = self.k_proj(hidden_states)
    
    if "v" in self.apply_module:
        v_guide_layer = self.v_proj if self.using_distorted_guide_layer else self.guide_layer.v_proj
        value_states = v_guide_layer(guide_hidden_states)
    else:
        value_states = self.v_proj(hidden_states)
    
    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def enable_llama_custom_attention(
    layer, 
    layer_id, 
    guide_layer=None, 
    method="extraction", 
    apply_module="qk", 
    top_k=1, 
    using_distorted_guide_layer=False,
    magnitude_dim="token",
):
    """
    replace the forward function of LlamaAttention with a custom forward function `llama_custom_attention_forward`
    """
    modified_module = layer.self_attn
    modified_module.layer_id = layer_id 
    
    if method == "extraction":
        modified_module.forward = types.MethodType(llama_custom_flash_attention_forward_4_40_extraction, modified_module)
    elif method == "apply":
        modified_module.guide_layer = guide_layer.self_attn
        modified_module.apply_module = apply_module
        modified_module.top_k = top_k
        modified_module.using_distorted_guide_layer = using_distorted_guide_layer
        modified_module.magnitude_dim = magnitude_dim
        assert magnitude_dim in ["token", "channel"]
        modified_module.forward = types.MethodType(llama_custom_flash_attention_forward_4_40_apply, modified_module)
    else:
        raise ValueError(f"method {method} not supported")

    return modified_module
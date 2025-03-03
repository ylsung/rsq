import torch
import torch.nn as nn

import yaml
import math
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
import kmean_utils


class InputWeightingModule:
    def __init__(self, model_type):
        self.batch_weighting = []
        
        if any(n in model_type.lower() for n in ["llama", "mistral", "qwen"]):
            self.model_type = "llama"
        else:
            raise ValueError(f"Unknown model type {model_type}")
        
    def __len__(self):
        return len(self.batch_weighting)

    def compute_weight(self, layer, input_tensor, output_tensor=None, **kwargs):
        raise NotImplementedError("This is an abstract method, need to be implemented in child class")

    def normalize_weight(self, input_tensor, min_value, max_value, quantile_value=None):
    
        if quantile_value is not None:
            q_min = 1 - quantile_value
            q_max = quantile_value
            # Ensure that q_max is the larger quantile
            if q_max < q_min:
                q_max, q_min = q_min, q_max
            quantile = torch.tensor([q_min, q_max]).to(input_tensor.device)
            tensor_min, tensor_max = torch.quantile(input_tensor, quantile)
        else: 
            tensor_min, tensor_max = torch.min(input_tensor), torch.max(input_tensor)
        normalized_input = (input_tensor - tensor_min) / (tensor_max - tensor_min)
        normalized_input = normalized_input * (max_value - min_value) + min_value
        normalized_input.clamp_(min_value, max_value)
        return normalized_input
    
    def bin_the_values(self, input_tensor, min_value, max_value, num_bins):
        # Ensure that A is a flattened tensor (to simplify operations)
        # Get quantile thresholds
        quantiles = torch.linspace(0, 1, num_bins+1)[1:-1].to(input_tensor.device)
        thresholds = torch.quantile(input_tensor.float(), quantiles)

        # Create a copy of the tensor to avoid in-place modifications
        result = input_tensor.clone()
        
        vlist = torch.linspace(min_value, max_value, num_bins)
        # Set values between each quantile range to the corresponding value in v
        for i in range(len(vlist)):
            if i == 0:
                mask = (input_tensor <= thresholds[i])
            elif i == len(vlist) - 1:
                mask = (input_tensor > thresholds[i-1])
            else:
                mask = (input_tensor > thresholds[i-1]) & (input_tensor <= thresholds[i])

            result[mask] = vlist[i]        # Set the corresponding value

        return result
            

def llama_original_attention(attn_module, hidden_states):
    attention_mask = None
    past_key_value = None

    # Custom function for getting a mask, modified from modeling_llama
    def get_causal_mask(
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
        
        return causal_mask

    # Get position ids and mask
    past_seen_tokens = 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
    )

    position_ids = cache_position.unsqueeze(0)

    attention_mask = get_causal_mask(attention_mask, hidden_states, cache_position, past_seen_tokens)

    # Computing attention weights
    bsz, q_len, _ = hidden_states.size()
    
    query_states = attn_module.q_proj(hidden_states)
    key_states = attn_module.k_proj(hidden_states)
    value_states = attn_module.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)

    past_key_value = getattr(attn_module, "past_key_value", past_key_value)
    cos, sin = attn_module.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states = repeat_kv(key_states, attn_module.num_key_value_groups)
    value_states = repeat_kv(value_states, attn_module.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn_module.head_dim)
    
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    return attn_weights


class OriginalAttentionWeighting(InputWeightingModule):
    def __init__(
        self, 
        model_type, 
        min_value=1, 
        max_value=3, 
        normalize="default", 
        scale=None, 
        num_bins=None, 
        masking=None, 
        quantile_value=None,
        truncate=None,
        **kwargs
    ):
        super().__init__(model_type)
        self.min_value = min_value
        self.max_value = max_value
        self.normalize = normalize
        self.scale = scale
        self.num_bins = num_bins
        self.masking = masking
        self.quantile_value = quantile_value
        self.truncate = truncate
        
        assert self.normalize in [None, "linear", "sqrt", "default"]
        
    def compute_weight(self, layer, input_tensor, output_tensor=None, **kwargs):
        
        if len(input_tensor.shape) == 2:
            # add the batch diemnsion
            input_tensor = input_tensor.unsqueeze(0)
        
        if self.model_type == "llama":
            attn_module = layer.self_attn
            # apply layernorm before attention
            input_tensor = layer.input_layernorm(input_tensor)
            
            past_seen_tokens = 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + input_tensor.shape[1], device=input_tensor.device
            )
            position_ids = cache_position.unsqueeze(0)
            # attn_weights = llama_original_attention(attn_module, input_tensor)
            attn_weights = attn_module(input_tensor, position_ids=position_ids, output_attentions=True)[1]
            
        weighting = attn_weights.sum(dim=1) # sum over heads
        weighting = weighting.sum(dim=1) # sum over tokens
        
        weighting = weighting.float()

        if self.scale == "square":
            weighting = weighting ** 2
        elif self.scale == "sqrt":
            weighting = weighting ** 0.5

        weighting = weighting.mean(dim=0) # mean over batch, but the batch size should be just one
        
        if self.normalize == "linear":
            tokens_used_the_token = torch.arange(0, len(weighting), device=weighting.device).flip(dims=[0]) + 1
            weighting = weighting / tokens_used_the_token
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value, quantile_value=self.quantile_value)
        elif self.normalize == "sqrt":
            tokens_used_the_token = torch.sqrt(torch.arange(0, len(weighting), device=weighting.device).flip(dims=[0]) + 1)
            weighting = weighting / tokens_used_the_token
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value, quantile_value=self.quantile_value)
        elif self.normalize == "default":
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value, quantile_value=self.quantile_value)
        
        if self.masking is not None:
            indices = weighting.topk(int(len(weighting) * self.masking), largest=False)[1]
            weighting = torch.ones_like(weighting)
            weighting[indices] = 0
        elif self.truncate is not None:
            indices = weighting.topk(int(len(weighting) * self.truncate), largest=False)[1]
            weighting[indices] = 0
        elif self.num_bins is not None:
            weighting = self.bin_the_values(weighting, self.min_value, self.max_value, self.num_bins)
        
        return weighting
        

class AdhocMaskingWeighting(InputWeightingModule):
    def __init__(self, model_type, method_type="first_half", **kwargs):
        super().__init__(model_type)
        self.method_type = method_type

    def compute_weight(self, layer, input_tensor, output_tensor=None, **kwargs):
        
        if len(input_tensor.shape) == 2:
            # add the batch diemnsion
            input_tensor = input_tensor.unsqueeze(0)

        weighting = torch.zeros(input_tensor.shape[1], device=input_tensor.device)

        if self.method_type == "first_half":
            weighting[input_tensor.shape[1]//2:] = 1
        elif self.method_type == "second_half":
            weighting[:input_tensor.shape[1]//2] = 1
        else:
            parts = [int(n) for n in self.method_type.split("_")]
            total_parts = parts.pop(-1)
            len_per_part = input_tensor.shape[1] // total_parts
            
            for part in parts:
                weighting[part*len_per_part:(part+1)*len_per_part] = 1
                
        return weighting

        
class MagnitudeWeighting(InputWeightingModule):
    def __init__(self, model_type, min_value=1, max_value=3, normalize="default", scale=None, num_bins=None, masking=None, input_or_output="input", reverse=False, dim=-1, truncate=None, **kwargs):
        super().__init__(model_type)
        self.min_value = min_value
        self.max_value = max_value
        self.normalize = normalize
        self.scale = scale
        self.num_bins = num_bins
        self.masking = masking
        self.input_or_output = input_or_output
        self.reverse = reverse
        self.dim = dim
        self.truncate = truncate
        
        assert self.normalize in [None, "linear", "sqrt", "default"]
        
    def compute_weight(self, layer, input_tensor, output_tensor=None, **kwargs):
        
        if len(input_tensor.shape) == 2:
            # add the batch diemnsion
            input_tensor = input_tensor.unsqueeze(0)
            output_tensor = output_tensor.unsqueeze(0)
        
        if self.input_or_output == "input":
            weighting = input_tensor.float().norm(dim=self.dim)
        elif self.input_or_output == "output":
            weighting = output_tensor.float().norm(dim=self.dim)
            
        if self.reverse:
            weighting = - weighting

        if self.scale == "square":
            weighting = weighting ** 2
        elif self.scale == "sqrt":
            weighting = weighting ** 0.5

        weighting = weighting.mean(dim=0) # mean over batch, but the batch size should be just one
        
        if self.normalize == "linear":
            tokens_used_the_token = torch.arange(0, len(weighting), device=weighting.device).flip(dims=[0]) + 1
            weighting = weighting / tokens_used_the_token
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        elif self.normalize == "sqrt":
            tokens_used_the_token = torch.sqrt(torch.arange(0, len(weighting), device=weighting.device).flip(dims=[0]) + 1)
            weighting = weighting / tokens_used_the_token
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        elif self.normalize == "default":
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        
        if self.masking is not None:
            indices = weighting.topk(int(len(weighting) * self.masking), largest=False)[1]
            weighting = torch.ones_like(weighting)
            weighting[indices] = 0
        elif self.truncate is not None:
            indices = weighting.topk(int(len(weighting) * self.truncate), largest=False)[1]
            weighting[indices] = 0
        elif self.num_bins is not None:
            weighting = self.bin_the_values(weighting, self.min_value, self.max_value, self.num_bins)

        return weighting
    
    
class ClusterWeighting(InputWeightingModule):
    def __init__(self, model_type, min_value=1, max_value=3, normalize="default", scale=None, num_bins=None, masking=None, input_or_output="input", reverse=False, dim=-1, truncate=None, n_clusters=100, **kwargs):
        super().__init__(model_type)
        self.min_value = min_value
        self.max_value = max_value
        self.normalize = normalize
        self.scale = scale
        self.num_bins = num_bins
        self.masking = masking
        self.input_or_output = input_or_output
        self.reverse = reverse
        self.dim = dim
        self.truncate = truncate
        self.n_clusters = n_clusters
        
        assert self.normalize in [None, "linear", "sqrt", "default"]
        
    def compute_weight(self, layer, input_tensor, output_tensor=None, **kwargs):
        # only support the case where the batch size is 1
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor[0]
            output_tensor = output_tensor[0]

        # output_tensor: (L, D)
        # cluster_embedding: (C, D)
        
        if self.input_or_output == "input":
            tensor_to_use = input_tensor.float()
        elif self.input_or_output == "output":
            tensor_to_use = output_tensor.float()
        
        _, cluster_embedding = kmean_utils.KMeans(tensor_to_use, K=self.n_clusters, Niter=30, verbose=False, using_loop=False)
        
        # D_ij = torch.zeros(tensor_to_use.shape[0], cluster_embedding.shape[0], device=tensor_to_use.device)
        # for i in range(D_ij.shape[1]):
        #     D_ij[:, i] = torch.sum((tensor_to_use - cluster_embedding[i])**2, dim=1)
            
        D_ij = - 2 * tensor_to_use.matmul(cluster_embedding.transpose(0, 1)) + \
            (tensor_to_use ** 2).sum(-1)[:, None] + \
            (cluster_embedding ** 2).sum(-1)[None, :]
            
        # D_ij = ((tensor_to_use.unsqueeze(1) - cluster_embedding.unsqueeze(0)) ** 2).sum(-1)  # (N, K) symbolic squared distances
        
        weighting = D_ij.min(dim=1)[0].view(-1)

        if self.scale == "square":
            weighting = weighting ** 2
        elif self.scale == "sqrt":
            weighting = weighting ** 0.5
            
        if self.reverse:
            weighting = - weighting

        if self.normalize == "linear":
            tokens_used_the_token = torch.arange(0, len(weighting), device=weighting.device).flip(dims=[0]) + 1
            weighting = weighting / tokens_used_the_token
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        elif self.normalize == "sqrt":
            tokens_used_the_token = torch.sqrt(torch.arange(0, len(weighting), device=weighting.device).flip(dims=[0]) + 1)
            weighting = weighting / tokens_used_the_token
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        elif self.normalize == "default":
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        
        if self.masking is not None:
            indices = weighting.topk(int(len(weighting) * self.masking), largest=False)[1]
            weighting = torch.ones_like(weighting)
            weighting[indices] = 0
        elif self.truncate is not None:
            indices = weighting.topk(int(len(weighting) * self.truncate), largest=False)[1]
            weighting[indices] = 0
        elif self.num_bins is not None:
            weighting = self.bin_the_values(weighting, self.min_value, self.max_value, self.num_bins)

        return weighting
    
    
class MaxDistWeighting(InputWeightingModule):
    def __init__(self, model_type, min_value=1, max_value=3, normalize="default", scale=None, num_bins=None, masking=None, input_or_output="input", reverse=False, **kwargs):
        super().__init__(model_type)
        self.min_value = min_value
        self.max_value = max_value
        self.normalize = normalize
        self.scale = scale
        self.num_bins = num_bins
        self.masking = masking
        self.input_or_output = input_or_output
        self.reverse = reverse
        
        assert self.normalize in [None, "linear", "sqrt", "default"]
        
    def compute_weight(self, layer, input_tensor, output_tensor=None, **kwargs):
        # only support the case where the batch size is 1
        if len(input_tensor.shape) == 3:
            # add the batch diemnsion
            input_tensor = input_tensor[0]
            output_tensor = output_tensor[0]
        
        if self.input_or_output == "input":
            tensor_to_use = input_tensor.float()
        elif self.input_or_output == "output":
            tensor_to_use = output_tensor.float()
        
        # (N, 1, dim) - (1, N, dim) = (N, N, dim) -> mean() -> (N, N)
        # dist = ((tensor_to_use.unsqueeze(1) - tensor_to_use.unsqueeze(0)) ** 2).sum(-1)
        
        dist = - 2 * tensor_to_use.matmul(tensor_to_use.transpose(0, 1)) + \
            (tensor_to_use ** 2).sum(-1)[:, None] + \
            (tensor_to_use ** 2).sum(-1)[None, :]
        
        weighting = dist.mean(dim=1).view(-1)

        if self.scale == "square":
            weighting = weighting ** 2
        elif self.scale == "sqrt":
            weighting = weighting ** 0.5
            
        if self.reverse:
            weighting = - weighting

        if self.normalize == "linear":
            tokens_used_the_token = torch.arange(0, len(weighting), device=weighting.device).flip(dims=[0]) + 1
            weighting = weighting / tokens_used_the_token
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        elif self.normalize == "sqrt":
            tokens_used_the_token = torch.sqrt(torch.arange(0, len(weighting), device=weighting.device).flip(dims=[0]) + 1)
            weighting = weighting / tokens_used_the_token
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        elif self.normalize == "default":
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        
        if self.masking is not None:
            indices = weighting.topk(int(len(weighting) * self.masking), largest=False)[1]
            weighting = torch.ones_like(weighting)
            weighting[indices] = 0
        
        elif self.num_bins is not None:
            weighting = self.bin_the_values(weighting, self.min_value, self.max_value, self.num_bins)
            
        return weighting
    
    
class MaxDiffWeighting(InputWeightingModule):
    def __init__(self, model_type, min_value=1, max_value=3, normalize="default", scale=None, num_bins=None, masking=None, input_or_output="input", reverse=False, **kwargs):
        super().__init__(model_type)
        self.min_value = min_value
        self.max_value = max_value
        self.normalize = normalize
        self.scale = scale
        self.num_bins = num_bins
        self.masking = masking
        self.input_or_output = input_or_output
        self.reverse = reverse
        
        assert self.normalize in [None, "linear", "sqrt", "default"]
        
    def compute_weight(self, layer, input_tensor, output_tensor=None, **kwargs):
        # only support the case where the batch size is 1
        if len(input_tensor.shape) == 3:
            # add the batch diemnsion
            input_tensor = input_tensor[0]
            output_tensor = output_tensor[0]

        # (N, 1, dim) - (1, N, dim) = (N, N, dim) -> mean() -> (N, N)
        # dist = ((tensor_to_use.unsqueeze(1) - tensor_to_use.unsqueeze(0)) ** 2).sum(-1)
        
        weighting = (input_tensor.float() - output_tensor.float()).norm(dim=-1).view(-1)
        
        if self.scale == "square":
            weighting = weighting ** 2
        elif self.scale == "sqrt":
            weighting = weighting ** 0.5
            
        if self.reverse:
            weighting = - weighting

        if self.normalize == "linear":
            tokens_used_the_token = torch.arange(0, len(weighting), device=weighting.device).flip(dims=[0]) + 1
            weighting = weighting / tokens_used_the_token
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        elif self.normalize == "sqrt":
            tokens_used_the_token = torch.sqrt(torch.arange(0, len(weighting), device=weighting.device).flip(dims=[0]) + 1)
            weighting = weighting / tokens_used_the_token
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        elif self.normalize == "default":
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        
        if self.masking is not None:
            indices = weighting.topk(int(len(weighting) * self.masking), largest=False)[1]
            weighting = torch.ones_like(weighting)
            weighting[indices] = 0
        
        elif self.num_bins is not None:
            weighting = self.bin_the_values(weighting, self.min_value, self.max_value, self.num_bins)
            
        return weighting
    
    
class TokenFreqWeighting(InputWeightingModule):
    def __init__(self, model_type, min_value=1, max_value=3, normalize="default", scale=None, num_bins=None, masking=None, input_or_output="input", reverse=False, **kwargs):
        super().__init__(model_type)
        self.min_value = min_value
        self.max_value = max_value
        self.normalize = normalize
        self.scale = scale
        self.num_bins = num_bins
        self.masking = masking
        self.input_or_output = input_or_output
        self.reverse = reverse
        
        assert self.normalize in [None, "linear", "sqrt", "default"]
        
    def compute_weight(self, layer, input_tensor, output_tensor=None, **kwargs):
        # only support the case where the batch size is 1
        token_freq = kwargs["token_freq"]

        # (N, 1, dim) - (1, N, dim) = (N, N, dim) -> mean() -> (N, N)
        # dist = ((tensor_to_use.unsqueeze(1) - tensor_to_use.unsqueeze(0)) ** 2).sum(-1)
        weighting = token_freq
        
        if self.scale == "square":
            weighting = weighting ** 2
        elif self.scale == "sqrt":
            weighting = weighting ** 0.5
            
        if self.reverse:
            weighting = - weighting

        if self.normalize == "linear":
            tokens_used_the_token = torch.arange(0, len(weighting), device=weighting.device).flip(dims=[0]) + 1
            weighting = weighting / tokens_used_the_token
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        elif self.normalize == "sqrt":
            tokens_used_the_token = torch.sqrt(torch.arange(0, len(weighting), device=weighting.device).flip(dims=[0]) + 1)
            weighting = weighting / tokens_used_the_token
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        elif self.normalize == "default":
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        
        if self.masking is not None:
            indices = weighting.topk(int(len(weighting) * self.masking), largest=False)[1]
            weighting = torch.ones_like(weighting)
            weighting[indices] = 0
        
        elif self.num_bins is not None:
            weighting = self.bin_the_values(weighting, self.min_value, self.max_value, self.num_bins)
            
        return weighting
    
    
class DotWeighting(InputWeightingModule):
    def __init__(self, model_type, min_value=1, max_value=3, normalize="default", scale=None, num_bins=None, masking=None, input_or_output="input", reverse=False, **kwargs):
        super().__init__(model_type)
        self.min_value = min_value
        self.max_value = max_value
        self.normalize = normalize
        self.scale = scale
        self.num_bins = num_bins
        self.masking = masking
        self.input_or_output = input_or_output
        self.reverse = reverse
        
        assert self.normalize in [None, "linear", "sqrt", "default"]
        
    def compute_weight(self, layer, input_tensor, output_tensor=None, **kwargs):
        
        if len(input_tensor.shape) == 2:
            # add the batch diemnsion
            input_tensor = input_tensor.unsqueeze(0)
            output_tensor = output_tensor.unsqueeze(0)
        
        if self.input_or_output == "input":
            input_tensor = input_tensor.float()
            weighting = input_tensor.bmm(input_tensor.transpose(1, 2)).sum(dim=-1)
        elif self.input_or_output == "output":
            output_tensor = output_tensor.float()
            weighting = output_tensor.bmm(output_tensor.transpose(1, 2)).sum(dim=-1)
        if self.reverse:
            weighting = - weighting

        if self.scale == "square":
            weighting = weighting ** 2
        elif self.scale == "sqrt":
            weighting = weighting ** 0.5

        weighting = weighting.mean(dim=0) # mean over batch, but the batch size should be just one
        
        if self.normalize == "linear":
            tokens_used_the_token = torch.arange(0, len(weighting), device=weighting.device).flip(dims=[0]) + 1
            weighting = weighting / tokens_used_the_token
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        elif self.normalize == "sqrt":
            tokens_used_the_token = torch.sqrt(torch.arange(0, len(weighting), device=weighting.device).flip(dims=[0]) + 1)
            weighting = weighting / tokens_used_the_token
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        elif self.normalize == "default":
            weighting = self.normalize_weight(weighting, min_value=self.min_value, max_value=self.max_value)
        
        if self.masking is not None:
            indices = weighting.topk(int(len(weighting) * self.masking), largest=False)[1]
            weighting = torch.ones_like(weighting)
            weighting[indices] = 0
        
        elif self.num_bins is not None:
            weighting = self.bin_the_values(weighting, self.min_value, self.max_value, self.num_bins)

        return weighting


def load_input_weighting_module(model_type, yaml_file_path, **kwargs):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    method_name = config['method_name']
    params = config['params']
    
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    params.update(**kwargs)
    
    try:
        return eval(method_name)(model_type=model_type, **params)

    except NameError:
        raise ValueError(f"Unknown module {method_name}")

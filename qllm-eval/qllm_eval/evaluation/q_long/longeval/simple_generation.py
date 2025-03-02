import torch
import inspect
import warnings
import copy

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.generation.logits_process import (
    LogitsProcessorList,
)

from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from transformers.utils import logging
from .modify_llama import enable_llama_custom_attention

logger = logging.get_logger(__name__)


@torch.no_grad()
def simple_generation(model, guide_model=None, guide_type=None, apply_module="qk", top_k=1.0, 
                      using_distorted_guide_layer=False, magnitude_dim="token", inputs=None, **kwargs):
    generation_config, model_kwargs = model._prepare_generation_config(None, **kwargs)

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        # logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id
    
    logits_processor = None
    stopping_criteria = None
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not model.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder:
        input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
        
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )
    
    model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
    
    prepared_stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )

    if guide_type == "weak_kv_strong_decoding":
        return weak_kv_strong_decoding(
            model,
            guide_model,
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=False,
            streamer=None,
            **model_kwargs,
        )
    elif guide_type == "attn_guide_decoding":
        return attn_guide_decoding(
            model,
            guide_model,
            apply_module,
            top_k,
            using_distorted_guide_layer,
            magnitude_dim,
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=False,
            streamer=None,
            **model_kwargs,
        )
    else:
        raise NotImplementedError(f"Guide type {guide_type} is not supported.")
        

def weak_kv_strong_decoding(
        model,
        guide_model,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ):
    this_peer_finished = False
    output_attentions = False
    output_hidden_states = False
    
    # # init values
    # logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    # stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    # if max_length is not None:
    #     warnings.warn(
    #         "`max_length` is deprecated in this function, use"
    #         " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
    #         UserWarning,
    #     )
    #     stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    # pad_token_id = pad_token_id if pad_token_id is not None else model.generation_config.pad_token_id
    # if eos_token_id is not None:
    #     logger.warning_once(
    #         "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
    #         " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
    #         " Otherwise make sure to set `model.generation_config.eos_token_id`",
    #         FutureWarning,
    #     )
    #     stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
    # else:
    #     # TODO remove when the method is totally private
    #     # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
    #     eos_token_id = [
    #         criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
    #     ]
    #     eos_token_id = eos_token_id[0] if eos_token_id else None
    #     if eos_token_id is None and model.generation_config.eos_token_id is not None:
    #         eos_token_id = model.generation_config.eos_token_id
    #         stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    
    batch_size, cur_len = input_ids.shape
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
    
    first_token = True
    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        # forward pass to get next token
        
        if first_token:
            outputs = model(**model_inputs, return_dict=True)
            first_token = False
        else:
            outputs = guide_model(**model_inputs, return_dict=True)
        
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itmodel is always small)
        next_token_logits = outputs.logits[:, -1, :].clone()

        next_tokens = torch.argmax(next_token_logits, dim=-1)

        # finished sentences should have their next token be a padding token
        # if has_eos_stopping_criteria:
        #     next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        
    return input_ids


def attn_guide_decoding(
        model,
        guide_model,
        apply_module,
        top_k,
        using_distorted_guide_layer,
        magnitude_dim,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ):
    this_peer_finished = False
    output_attentions = False
    output_hidden_states = False
    
    # # init values
    # logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    # stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    # if max_length is not None:
    #     warnings.warn(
    #         "`max_length` is deprecated in this function, use"
    #         " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
    #         UserWarning,
    #     )
    #     stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    # pad_token_id = pad_token_id if pad_token_id is not None else model.generation_config.pad_token_id
    # if eos_token_id is not None:
    #     logger.warning_once(
    #         "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
    #         " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
    #         " Otherwise make sure to set `model.generation_config.eos_token_id`",
    #         FutureWarning,
    #     )
    #     stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
    # else:
    #     # TODO remove when the method is totally private
    #     # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
    #     eos_token_id = [
    #         criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
    #     ]
    #     eos_token_id = eos_token_id[0] if eos_token_id else None
    #     if eos_token_id is None and model.generation_config.eos_token_id is not None:
    #         eos_token_id = model.generation_config.eos_token_id
    #         stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    
    batch_size, cur_len = input_ids.shape
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
    
    guide_model_kwargs = copy.deepcopy(model_kwargs)

    # only work for llama for now
    for layer_id, (layer, guide_layer) in enumerate(zip(model.model.layers, guide_model.model.layers)):
        enable_llama_custom_attention(
            layer, layer_id, guide_layer=guide_layer, apply_module=apply_module, method="apply", top_k=top_k,
            using_distorted_guide_layer=using_distorted_guide_layer,
            magnitude_dim=magnitude_dim,
        )
        enable_llama_custom_attention(guide_layer, layer_id, guide_layer=None, method="extraction")
    
    first_token = True
    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        guide_model_inputs = guide_model.prepare_inputs_for_generation(input_ids, **guide_model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
        
        guide_model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        guide_model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        # to cache the perfect attns from the guide model
        guide_model_outputs = guide_model(**guide_model_inputs, return_dict=True)
        
        # forward pass to get next token with the cached attn
        outputs = model(**model_inputs, return_dict=True)
        
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itmodel is always small)
        next_token_logits = outputs.logits[:, -1, :].clone()

        next_tokens = torch.argmax(next_token_logits, dim=-1)

        # finished sentences should have their next token be a padding token
        # if has_eos_stopping_criteria:
        #     next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        
        guide_model_kwargs = guide_model._update_model_kwargs_for_generation(
            guide_model_outputs,
            guide_model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1
        
        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        
    return input_ids


def greedy_decoding(
        model,
        guide_model,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ):
    this_peer_finished = False
    output_attentions = False
    output_hidden_states = False
    
    # # init values
    # logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    # stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    # if max_length is not None:
    #     warnings.warn(
    #         "`max_length` is deprecated in this function, use"
    #         " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
    #         UserWarning,
    #     )
    #     stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    # pad_token_id = pad_token_id if pad_token_id is not None else model.generation_config.pad_token_id
    # if eos_token_id is not None:
    #     logger.warning_once(
    #         "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
    #         " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
    #         " Otherwise make sure to set `model.generation_config.eos_token_id`",
    #         FutureWarning,
    #     )
    #     stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
    # else:
    #     # TODO remove when the method is totally private
    #     # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
    #     eos_token_id = [
    #         criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
    #     ]
    #     eos_token_id = eos_token_id[0] if eos_token_id else None
    #     if eos_token_id is None and model.generation_config.eos_token_id is not None:
    #         eos_token_id = model.generation_config.eos_token_id
    #         stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    
    batch_size, cur_len = input_ids.shape
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        # forward pass to get next token
        outputs = model(**model_inputs, return_dict=True)
        
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itmodel is always small)
        next_token_logits = outputs.logits[:, -1, :].clone()

        next_tokens = torch.argmax(next_token_logits, dim=-1)

        # finished sentences should have their next token be a padding token
        # if has_eos_stopping_criteria:
        #     next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        
    return input_ids

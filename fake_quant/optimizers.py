import yaml
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from copy import deepcopy
import warnings
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Union, Tuple
import logging


def iterate_minibatches(
    *tensors: torch.Tensor,
    batch_size: int,
    allow_incomplete: bool = True,
    device: Optional[torch.device] = None,
    callback: Callable[[Sequence[torch.Tensor]], Sequence[torch.Tensor]] = lambda x: x,
) -> Iterator[Sequence[torch.Tensor]]:
    # borrow from https://github.com/Vahe1994/AQLM/blob/main/src/utils.py
    """
    Samples data points *forever*, in random order, with less overhead than DataLoader;
    Adapted from https://github.com/stanis-morozov/unq/blob/master/lib/utils.py
    probably implemented over9000 times in transformers, torch, etc
    :param tensors: one or more tensors with the same 0-th dimension
    :param batch_size: sample this many points with each yield
    :param allow_incomplete: if True and if dataset size is not divisible by batch size, the last batch
        may have less than :batch_size: samples to cover the entire dataset. If False, the last batch is dropped
    :param callback: optional function to be called on each batch of tensors before it is yielded to the user
    :returns: generates a tuple of minibatches from each tensor, same length as input *tensors
        If a batch contains only one tensor, this function will yield a tensor (and not a tuple/list with one tensor)
    """
    num_samples = len(tensors[0])
    assert all(len(x) == num_samples for x in tensors)
    indices = torch.randperm(num_samples, device=tensors[0].device)
    
    while True:
        prev_batch = None
        for batch_start in range(0, len(indices), batch_size):
            if not allow_incomplete and batch_start + batch_size > len(indices):
                break
            batch_ix = indices[batch_start : batch_start + batch_size]
            batch = callback(tuple(tensor[batch_ix].to(device, non_blocking=True) for tensor in tensors))
            if prev_batch is not None:
                yield prev_batch
            prev_batch = batch if isinstance(batch, (list, tuple)) and len(tensors) > 1 else batch[0]
            del batch
        yield prev_batch


def _compute_mse_on_batch(
    layer: nn.Module, batch_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]], **kwargs
) -> torch.Tensor:
    # borrowed from https://github.com/Vahe1994/AQLM/blob/main/src/finetune.py
    """
    Compute the activation MSE error between transformer layers
    :param
    """
    inps_batch, outs_batch = next(batch_iter)
    # inps_batch = inps_batch.to(dtype=torch.float32)
    # outs_batch = outs_batch.to(dtype=torch.float32)

    if inps_batch.shape[0] != 1:  # replicate kwargs to match the batch size
        for name, value in list(kwargs.items()):
            if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                if name not in ("attention_mask", "position_ids"):
                    warnings.warn(f"Tiling an unexpected kwarg {name} over batch size; make sure this is valid.")
                repeats = [len(inps_batch)] + [1 for _ in range(value.ndim - 1)]
                kwargs[name] = value.tile(*repeats)

    outs_prediction, *_unused = layer(inps_batch, **kwargs)
    assert outs_prediction.shape == outs_batch.shape
    return F.mse_loss(outs_prediction, outs_batch)


def get_prob_loss(predict_attn_weights, target_attn_weights, sequential=False):
    divergence = torch.nn.KLDivLoss(reduction="sum", log_target=True)
    
    attn_loss = 0
    if sequential:
        # iteratively compute the loss over batch size and head size
        for j in range(predict_attn_weights.shape[1]):
            predict_attn_weights_j = torch.nn.functional.log_softmax(predict_attn_weights[:, j], dim=-1, dtype=torch.float32)
            target_attn_weights_j = torch.nn.functional.log_softmax(target_attn_weights[:, j], dim=-1, dtype=torch.float32)
            attn_loss += divergence(predict_attn_weights_j, target_attn_weights_j)
    else:
        predict_attn_weights = torch.nn.functional.log_softmax(predict_attn_weights, dim=-1, dtype=torch.float32)
        target_attn_weights = torch.nn.functional.log_softmax(target_attn_weights, dim=-1, dtype=torch.float32)
        attn_loss = divergence(predict_attn_weights, target_attn_weights)

    batch_dim = torch.numel(target_attn_weights) / target_attn_weights.shape[-1]
    attn_loss = attn_loss / batch_dim
    
    return attn_loss


def _compute_loss_on_batch(
    layer: nn.Module, 
    dev: torch.device, 
    args: Any, 
    batch_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]], **kwargs
) -> torch.Tensor:
    # borrowed from https://github.com/Vahe1994/AQLM/blob/main/src/finetune.py
    """
    Compute the activation MSE error between transformer layers
    :param
    """
    inps_batch, outs_batch, clean_inps_batch, clean_outs_batch = next(batch_iter)
    # inps_batch = inps_batch.to(dtype=torch.float32)
    # outs_batch = outs_batch.to(dtype=torch.float32)
    inps_batch = inps_batch.to(dev)
    outs_batch = outs_batch.to(dev)
    clean_inps_batch = clean_inps_batch.to(dev)
    clean_outs_batch = clean_outs_batch.to(dev)
    
    org_attn_module = kwargs.pop("org_attn_module", None)
    quant_attn_module = kwargs.pop("quant_attn_module", None)

    if inps_batch.shape[0] != 1:  # replicate kwargs to match the batch size
        for name, value in list(kwargs.items()):
            if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                if name not in ("attention_mask", "position_ids"):
                    warnings.warn(f"Tiling an unexpected kwarg {name} over batch size; make sure this is valid.")
                repeats = [len(inps_batch)] + [1 for _ in range(value.ndim - 1)]
                kwargs[name] = value.tile(*repeats)

    outs_prediction, *_unused = layer(inps_batch, **kwargs)
    assert outs_prediction.shape == outs_batch.shape
    
    mse_loss = F.mse_loss(
        outs_prediction, 
        outs_batch if not args.clean_outs_for_mse else clean_outs_batch,
    )

    attn_loss = torch.tensor(0).to(mse_loss.device)
    
    if args.compute_self_similarity_loss:
        target_outs_batch = outs_batch if not args.clean_outs_for_attn_loss else clean_outs_batch
        target_similarity = target_outs_batch.bmm(target_outs_batch.transpose(1, 2))
        predict_similarity = outs_prediction.bmm(outs_prediction.transpose(1, 2))
        
        mask = torch.triu(target_similarity, diagonal=1) > 0 # symmtric, only take half of the similarities
        attn_loss = F.mse_loss(predict_similarity[mask], target_similarity[mask])
    
    # compute the attn loss
    if args.compute_attn_loss:
        target_attn_weights = org_attn_module(
            inps_batch if not args.clean_outs_for_attn_loss else clean_inps_batch, 
            **kwargs).detach() # do not backpropagate through the original attn module
        predict_attn_weights = quant_attn_module(inps_batch, **kwargs)
        
        if args.attn_loss_on_prob:
            attn_loss = get_prob_loss(predict_attn_weights, target_attn_weights, sequential=args.attn_loss_on_prob_sequential)
        else:
            mask = target_attn_weights > - 10 ** 10 # only take the valid causal attention weights
            attn_loss = F.mse_loss(predict_attn_weights[mask], target_attn_weights[mask])
            
    if args.compute_next_attn_loss and org_attn_module is not None:
        target_attn_weights = org_attn_module(
            outs_batch if not args.clean_outs_for_attn_loss else clean_outs_batch, 
            **kwargs).detach() # do not backpropagate through the original attn module
        predict_attn_weights = org_attn_module(outs_prediction, **kwargs)
        
        if args.attn_loss_on_prob:
            attn_loss = get_prob_loss(predict_attn_weights, target_attn_weights, sequential=args.attn_loss_on_prob_sequential)
        else:
            mask = target_attn_weights > - 10 ** 10 # only take the valid causal attention weights
            attn_loss = F.mse_loss(predict_attn_weights[mask], target_attn_weights[mask])
            
    return mse_loss, attn_loss


class GradientOptimizer:
    def __init__(
        self, 
        quant_parameters_requires_grad="scale|zero",
        weight_parameters_requires_grad="weight_fp",
        verbose=False, 
        **kwargs,
    ):
        opt_params = {
            "quant_lr": 1e-4,
            "weight_lr": 1e-5,
            "finetune_max_epochs": 10, 
            "finetune_early_stop": 3,
            "local_batch_size": 1,
            "finetune_batch_size": 32,
            "attn_loss_weight": 1,
            "attn_loss_ratio": None,
        }
        
        for k, v in kwargs.items():
            if k in opt_params:
                # overwrite the values if specifiying
                opt_params[k] = v
        
        self.opt_params = opt_params
        self.verbose = verbose
        self.quant_parameters_requires_grad = quant_parameters_requires_grad
        self.weight_parameters_requires_grad = weight_parameters_requires_grad
        
        if self.verbose:
            logging.info("Optimizer parameters:")
            logging.info(self.opt_params)

    @torch.enable_grad()
    def finetune(
        self,
        layer,
        dev,
        args,
        train_inps,
        train_outs,
        train_clean_inps,
        train_clean_outs,
        val_inps,
        val_outs,
        val_clean_inps,
        val_clean_outs,
        **kwargs
    ):
        if train_clean_inps is None:
            train_clean_inps = train_inps
        if train_clean_outs is None:
            train_clean_outs = train_outs
        if val_clean_inps is None:
            val_clean_inps = val_inps
        if val_clean_outs is None:
            val_clean_outs = val_outs

        weight_lr = self.opt_params["weight_lr"]
        quant_lr = self.opt_params["quant_lr"]
        finetune_max_epochs = self.opt_params["finetune_max_epochs"]
        finetune_early_stop = self.opt_params["finetune_early_stop"]
        local_batch_size = self.opt_params["local_batch_size"]
        finetune_batch_size = self.opt_params["finetune_batch_size"]
        attn_loss_weight = self.opt_params["attn_loss_weight"]
        attn_loss_ratio = self.opt_params["attn_loss_ratio"]
        
        differentiable_parameters_by_name = {}
        quant_parameters_requires_grad = self.quant_parameters_requires_grad.split("|")
        weight_parameters_requires_grad = self.weight_parameters_requires_grad.split("|")
        quant_params = []
        weight_params = []
        for n, p in layer.named_parameters():
            if any([r in n for r in quant_parameters_requires_grad]):
                p.requires_grad = True
                differentiable_parameters_by_name[n] = p
                quant_params.append(p)
            elif any([r in n for r in weight_parameters_requires_grad]):
                p.requires_grad = True
                differentiable_parameters_by_name[n] = p
                weight_params.append(p)
            else:
                p.requires_grad = False
                
        # print(list(differentiable_parameters_by_name.values())[:3])
        
        print(differentiable_parameters_by_name.keys())
        
        opt = torch.optim.AdamW([
            {"params": weight_params, "lr": weight_lr},
            {"params": quant_params, "lr": quant_lr}], 
            lr=weight_lr, betas=(0.9, 0.99), eps=1e-06, weight_decay=0.
        )

        num_samples = len(train_inps)

        num_accumulation_steps = finetune_batch_size // local_batch_size

        train_batches_per_epoch = num_samples // local_batch_size
        train_batch_iterators = [
            iterate_minibatches(
                train_inps, 
                train_outs, 
                train_clean_inps,
                train_clean_outs,
                batch_size=local_batch_size, 
                device=train_inps.device,
            )
            for i in range(1)
        ]

        run_validation = False
        if val_inps is not None and val_outs is not None:
            run_validation = True
            valid_batches_per_epoch = len(val_inps) // local_batch_size
            valid_batch_iterators = [
                iterate_minibatches(
                    val_inps, 
                    val_outs, 
                    val_clean_inps, 
                    val_clean_outs, 
                    batch_size=local_batch_size, 
                    device=val_inps.device
                )
                for i in range(1)
            ]
            
        if run_validation:
            # evaluate before training
            layer.eval()
            loss_numerator = loss_denominator = 0
            mse_loss_numerator = 0
            attn_loss_numerator = 0
            with torch.no_grad():
                for _ in range(valid_batches_per_epoch):
                    mse_loss, attn_loss = _compute_loss_on_batch(layer, dev, args, valid_batch_iterators[0], **kwargs)
                    mse_loss_numerator += mse_loss.item()
                    attn_loss_numerator += attn_loss.item()
                    loss_denominator += 1
            mse_loss_epoch = mse_loss_numerator / loss_denominator
            attn_loss_epoch = attn_loss_numerator / loss_denominator
            
            if attn_loss_ratio is not None and attn_loss_epoch != 0:
                attn_loss_weight = attn_loss_ratio / attn_loss_epoch * mse_loss_epoch
                logging.info(f"attn_loss_weight is updated to {attn_loss_weight:.2e} to achieve the ratio {attn_loss_ratio}")
            
            valid_loss_epoch = mse_loss_epoch + attn_loss_weight * attn_loss_epoch
            logging.info(f"Evaluation before training ({valid_batches_per_epoch * local_batch_size} samples).")
            logging.info(f"valid loss={valid_loss_epoch:.2e}\tmse_loss={mse_loss_epoch:.2e}\tattn_loss={attn_loss_epoch:.2e}")
            best_loss = valid_loss_epoch
            best_parameters_by_name = deepcopy(differentiable_parameters_by_name)
            worse_count = 0

        steps_accumulated = 0
        for epoch in range(finetune_max_epochs):
            layer.train()
            # train epoch
            loss_numerator = loss_denominator = 0
            mse_loss_numerator = 0
            attn_loss_numerator = 0
            for _ in range(train_batches_per_epoch):
                mse_loss, attn_loss = _compute_loss_on_batch(layer, dev, args, train_batch_iterators[0], **kwargs)
                loss = mse_loss + attn_loss_weight * attn_loss
                mse_loss_numerator += mse_loss.item()
                attn_loss_numerator += attn_loss.item()
                
                (loss / num_accumulation_steps).backward()
                steps_accumulated += 1

                if not torch.isfinite(loss).item():
                    raise ValueError(f"Fine-tuning loss is {loss}")

                if steps_accumulated >= num_accumulation_steps:
                    opt.step()
                    opt.zero_grad()
                    steps_accumulated = 0

                loss_numerator += loss.item()
                loss_denominator += 1
            train_loss_epoch = loss_numerator / loss_denominator
            train_mse_loss_epoch = mse_loss_numerator / loss_denominator
            train_attn_loss_epoch = attn_loss_numerator / loss_denominator
            if run_validation:
                layer.eval()
                # val epoch
                loss_numerator = loss_denominator = 0
                mse_loss_numerator = 0
                attn_loss_numerator = 0
                with torch.no_grad():
                    for _ in range(valid_batches_per_epoch):
                        mse_loss, attn_loss = _compute_loss_on_batch(layer, dev, args, valid_batch_iterators[0], **kwargs)
                        loss = mse_loss + attn_loss_weight * attn_loss
                        mse_loss_numerator += mse_loss.item()
                        attn_loss_numerator += attn_loss.item()

                        loss_numerator += loss.item()
                        loss_denominator += 1
                valid_loss_epoch = loss_numerator / loss_denominator
                valid_mse_loss_epoch = mse_loss_numerator / loss_denominator
                valid_attn_loss_epoch = attn_loss_numerator / loss_denominator
                
            # log losses in the end of the epoch
            if self.verbose:
                logging.info("-" * 10)
                logging.info(f"epoch={epoch}")
                logging.info(f"train loss={train_loss_epoch:.2e}\tmse_loss={train_mse_loss_epoch:.2e}\tattn_loss={train_attn_loss_epoch:.2e}")
                if run_validation:
                    logging.info(f"valid loss={valid_loss_epoch:.2e}\tmse_loss={valid_mse_loss_epoch:.2e}\tattn_loss={valid_attn_loss_epoch:.2e}")

            if run_validation:
                if valid_loss_epoch < best_loss:
                    logging.info(f"new best loss {valid_loss_epoch:.2e} on epoch {epoch}")
                    best_loss = valid_loss_epoch
                    best_parameters_by_name = deepcopy(differentiable_parameters_by_name)
                    worse_count = 0
                else:
                    worse_count += 1
                    if worse_count >= finetune_early_stop:
                        break

        if run_validation:
            layer.load_state_dict(best_parameters_by_name, strict=False)
            
        # print(list(differentiable_parameters_by_name.values())[:3])

        return layer


def load_optimizer(yaml_file_path, **kwargs):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    method_name = config['method_name']
    params = config['params']
    
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    params.update(**kwargs)
    
    try:
        return eval(method_name)(**params)

    except NameError:
        raise ValueError(f"Unknown scheduler {method_name}")
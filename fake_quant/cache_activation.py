import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import eval_utils
import hadamard_utils
import logging
import os
import time

from api import load_quantized_checkpoint


def main():
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)
        wandb.run.name = args.save_name
        
    utils.config_logging(os.path.join(args.save_path, f'{args.save_name}.log'))
    
    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token, args.use_flash_attn)
    model.eval()

    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if 'down_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                # TODO: check if we can merge these two
                if "mistral" in args.model:
                    qlayers[name].had_dim = model.config.head_dim
                else:
                    qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present
    
    quantization_time_start = time.time()

    logging.info(f"quantize with {args.nsamples} of length {args.train_seqlen} {args.cal_dataset} samples")
    trainloader = data_utils.get_loaders(
        args.cal_dataset, nsamples=args.nsamples,
        seed=args.seed, model=args.model,
        seqlen=args.train_seqlen, eval_mode=False
    )
    
    from collections import defaultdict
    grad_activation = defaultdict(list)

    def save_activation(name):
        def hook(model, grad_input, grad_output):
            grad_activation[name].append(
                grad_output.detach().cpu()
            )
        return hook
    
    os.makedirs(args.activation_folder, exist_ok=True)
    
    import pdb; pdb.set_trace()
    
    for layer_id in range(model.config.num_hidden_layers):
        model.model.layers[layer_id].register_module_backward_hook(save_activation(f'layer_{layer_id}'))
        
    for d in trainloader:
        inputs = d[0].cuda()

if __name__ == '__main__':
    main()
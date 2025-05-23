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

    if args.load_qmodel_path:
        # load quantized weights
        print(f"load rotated: {args.load_qmodel_path}")
        model = load_quantized_checkpoint(
            model, args.load_qmodel_path, rotate=args.rotate
        )
        
        model = model.eval()
        # model.cuda()
    else:
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
        save_dict = {}
        if args.w_bits < 16:
            if args.load_qmodel_path: # Load Quantized Rotated Model
                # assert args.rotate, "Model should be rotated to load a quantized model!"
                assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
                # logging.info("Load quantized model from ", args.load_qmodel_path) # has some weird errors
                print("Load quantized model from ", args.load_qmodel_path)
                save_dict = torch.load(args.load_qmodel_path)
                model.load_state_dict(save_dict["model"])
                
            elif not args.w_rtn: # GPTQ Weight Quantization
                # assert "llama" in args.model, "Only llama is supported for GPTQ!"
                
                logging.info(f"quantize with {args.nsamples} of length {args.train_seqlen} {args.cal_dataset} samples")
                trainloader = data_utils.get_loaders(
                    args.cal_dataset, nsamples=args.nsamples,
                    seed=args.seed, model=args.model,
                    seqlen=args.train_seqlen, eval_mode=False
                )
                
                if args.expand_factor > 1:
                    trainloader = data_utils.expand_dataset(trainloader, args.expand_factor)

                quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
                save_dict["w_quantizers"] = quantizers
            else: # RTN Weight Quantization
                quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
                save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)
        
        torch.cuda.synchronize()
        logging.info(f"quantization time: {time.time() - quantization_time_start}")
        

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
        
        for name in qlayers:            
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not(args.a_asym)
            layer_a_clip = args.a_clip_ratio
            
            if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                              groupsize=args.v_groupsize,
                                              sym=not(args.v_asym),
                                              clip_ratio=args.v_clip_ratio)
            
            if 'lm_head' in name: #Skip lm_head quantization   
                layer_input_bits = 16
            
            if 'down_proj' in name: #Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

                
            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip)

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                          "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn, 
                            rope_function_name, 
                            config=model.config,
                            **k_quant_config)
        
    # Evaluating on dataset
    testloader = data_utils.get_loaders(
            args.eval_dataset,
            seed=args.seed,
            model=args.model,
            seqlen=args.val_seqlen, #model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True
        )

    
    if not args.skip_wiki_eval:
        dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
        if args.wandb:
            wandb.log({'ppl/{}'.format(args.eval_dataset.upper()): dataset_ppl})

    if not args.lm_eval:
        return
    else:
        # Import lm_eval utils
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM

    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    hflm = HFLM(
        pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size,
        max_length=args.lm_eval_max_length,
        # device=args.lm_eval_device
        )
    
    utils.cleanup_memory(verbos=True)
    
    # task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
    task_names = args.tasks
    results = lm_eval.simple_evaluate(
        hflm, 
        tasks=task_names, 
        num_fewshot=args.num_fewshot, 
        batch_size=args.lm_eval_batch_size,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        limit=args.limit, # for testing
    )

    results = results['results']
    
    def get_number(result):
        
        keys_order = [
            "acc_norm,none",
            "acc,none",
            "exact_match,flexible-extract",
            "exact_match,none",
            "exact,none",
            "exact_match,get-answer", # For BBH
            "exact_match,remove_whitespace", # For trivia QA
        ]
        
        for key in keys_order:
            if key in result:
                return result[key]

        raise NotImplementedError("No metric found in result")

    metric_vals = {task: round(get_number(result), 4) for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    logging.info(metric_vals)

    if args.wandb:
        wandb.log(metric_vals)


if __name__ == '__main__':
    main()
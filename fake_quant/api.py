import torch

import rotation_utils  
import hadamard_utils
import quant_utils
import utils


def load_quantized_checkpoint(model, checkpoint, rotate=False):
    class ARGS:
        rotate_mode = "hadamard"
        fp32_had = False

    args = ARGS()
    # Rotate the weights
    if rotate:
        rotation_utils.fuse_layer_norms(model)
        # comment this line out because it doesn't affect anything if we load the weight after
        # rotation_utils.rotate_model(model, args)
        rotation_utils.post_process_model_after_load(model, args)
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
                # qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                if model.config.model_type in ("mistral"):
                    qlayers[name].had_dim = model.config.head_dim
                else:
                    qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present

    save_dict = torch.load(checkpoint)
    model.load_state_dict(save_dict["model"])
    
    return model


def rotate(model):
    class ARGS:
        rotate_mode = "hadamard"
        fp32_had = False

    args = ARGS()
    # Rotate the weights
    rotation_utils.fuse_layer_norms(model)
    # comment this line out because it doesn't affect anything if we load the weight after
    rotation_utils.rotate_model(model, args)
    # rotation_utils.post_process_model_after_load(model, args)
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
            qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
            qlayers[name].fp32_had = args.fp32_had

    return model
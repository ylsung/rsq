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
from collections import defaultdict

# def add_batch(name):
#     def tmp(_, inp, out):
#         gptq[name].add_batch(inp[0].data, out.data)
#     return tmp
# handles = []
# for name in subset:
#     handles.append(subset[name].register_forward_hook(add_batch(name)))
# for j in range(args.nsamples):
#     outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
# for h in handles:
#     h.remove()

def main():
    args = utils.parser_gen()
        
    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    
    # Rotate the weights
    if args.rotate:
        print("Rotate")
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
                qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
                
    trainloader = data_utils.get_loaders(
        args.cal_dataset, nsamples=args.nsamples,
        seed=args.seed, model=args.model,
        seqlen=model.seqlen, eval_mode=False
    )
    
    
    model.cuda()
    qlayers = quant_utils.find_qlayers(model)
    
    input_acts = defaultdict(list)
    output_acts = defaultdict(list)
    def add_batch(name):
        def tmp(_, inp, out):
            input_acts[name].append(inp[0].data.cpu())
            output_acts[name].append(out.data.cpu())
        return tmp

    handles = []
    for name, layer in qlayers.items():
        handles.append(layer.register_forward_hook(add_batch(name)))

    for i, d in enumerate(trainloader):
        print(i)
        if i >= 4:
            break
        outputs = model(d[0].cuda())
        
    
    if args.rotate:
        torch.save(input_acts, "inputs/after_rotate.pth")
        torch.save(output_acts, "outputs/after_rotate.pth")
        torch.save(model.state_dict(), "weights/after_rotate.pth")
    else:
        torch.save(input_acts, "inputs/before_rotate.pth")
        torch.save(output_acts, "outputs/before_rotate.pth")
        torch.save(model.state_dict(), "weights/before_rotate.pth")

# before rotate: tensor([[7.5137, 6.8937, 7.8090,  ..., 2.2511, 1.8989, 2.6839]]
# after rotate : tensor([[7.5284, 6.8921, 7.8058,  ..., 2.2544, 1.8972, 2.6808]]

if __name__ == '__main__':
    main()
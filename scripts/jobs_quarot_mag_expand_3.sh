source /home/ylsung/codes/env.sh

nsamples_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
train_seqlen_list=(1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024 512 256)

# 3 to 12
# 3 6 9 12
nsamples_list=(128 256 512 1024 2048 8192 4096 16384 32768)
train_seqlen_list=(8192 4096 2048 1024 512 128 256 64 32)
val_size_list=(16 32 64 128 256 1024 512 2048 4096)

declare -a jobs

seeds=(0 1 2)

source scripts/additional_eval_func.sh

i=2

num_bins=0.01
attn_weighting=org_attn_none


nsamples=${nsamples_list[i]}
train_seqlen=${train_seqlen_list[i]}
val_size=${val_size_list[i]}

# for dataset in ptb
# do
#     nsamples=${nsamples_list[i]}
#     train_seqlen=${train_seqlen_list[i]}
#     val_size=${val_size_list[i]}
#     for w_bits in 3
#     do
#         for seed in 2
#         do
#             q_type=${dataset}_quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}1_min${num_bins}_expand8@${seed}
#             save_name=llama3-8b-instruct_${q_type}
#             # no rotate

#             job="eval cd /home/ylsung/codes/QuaRot/; \
#             python fake_quant/main.py \
#             --model meta-llama/Meta-Llama-3-8B-Instruct \
#             --w_bits ${w_bits} --w_clip \
#             --rotate \
#             --seed ${seed} \
#             --cal_dataset ${dataset} \
#             --min_value ${num_bins} \
#             --max_value 1 \
#             --add_until_fail \
#             --expand_factor 8 \
#             --offload_activations \
#             --val_size ${val_size} \
#             --module_input_weighting_yaml fake_quant/configs/input_weighting/${attn_weighting}3.yaml \
#             --nsamples ${nsamples} \
#             --train_seqlen ${train_seqlen} \
#             --wandb_project QuaRot \
#             --wandb_id ylsung \
#             --save_name ${save_name} \
#             --lm_eval \
#             --save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
#             --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

#             echo $job
#             jobs+=("$job")
#         done
#     done
# done

# for w_bits in 3
# do
#     for seed in 0 1 2
#     do
#         q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}1_min${num_bins}_expand8@${seed}
#         save_name=qwen-2.5-32B-instruct_${q_type}
#         # no rotate

#         job="eval cd /home/ylsung/codes/QuaRot/; \
#         python fake_quant/main.py \
#         --model Qwen/Qwen2.5-32B-Instruct \
#         --w_bits ${w_bits} --w_clip \
#         --rotate \
#         --seed ${seed} \
#         --min_value ${num_bins} \
#         --max_value 1 \
#         --add_until_fail \
#         --expand_factor 8 \
#         --offload_activations \
#         --val_size ${val_size} \
#         --module_input_weighting_yaml fake_quant/configs/input_weighting/${attn_weighting}3.yaml \
#         --nsamples ${nsamples} \
#         --train_seqlen ${train_seqlen} \
#         --wandb_project QuaRot \
#         --wandb_id ylsung \
#         --save_name ${save_name} \
#         --lm_eval \
#         --save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
#         --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

#         echo $job
#         jobs+=("$job")
#     done
# done


# for dataset in c4 pajama ptb
# do
#     nsamples=${nsamples_list[i]}
#     train_seqlen=${train_seqlen_list[i]}
#     val_size=${val_size_list[i]}
#     for w_bits in 3
#     do
#         for seed in 0 1 2
#         do
#             q_type=${dataset}_quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}1_min${num_bins}_expand8@${seed}
#             save_name=llama3-8b-instruct_${q_type}
#             # no rotate
            
#             mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name})

#             for item in "${new_jobs[@]}"; do
#                 echo $item
#             done

#             jobs+=("${new_jobs[@]}")
#         done
#     done
# done


# nsamples=${nsamples_list[i]}
# train_seqlen=${train_seqlen_list[i]}
# val_size=${val_size_list[i]}
# for w_bits in 2 4
# do
#     for seed in 0 1 2
#     do
#         q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}1_min${num_bins}_expand8@${seed}
#         save_name=llama3-8b-instruct_${q_type}
#         # no rotate
        
#         mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name})

#         for item in "${new_jobs[@]}"; do
#             echo $item
#         done

#         jobs+=("${new_jobs[@]}")
#     done
# done


# for w_bits in 3
# do
#     for seed in 0 1 2
#     do
#         q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}1_min${num_bins}_expand8@${seed}
#         save_name=mistral-nemo-instruct-2407_${q_type}
#         # no rotate
        
#         mapfile new_jobs < <(add_additional_jobs "mistralai/Mistral-Nemo-Instruct-2407" "mistral-nemo-instruct-2407" ${save_name})

#         for item in "${new_jobs[@]}"; do
#             echo $item
#         done

#         jobs+=("${new_jobs[@]}")
#     done
# done


# source scripts/additional_eval_func_qwen.sh

# for model_size in 7 14 32
# do
#     for w_bits in 3
#     do
#         for seed in 0 1 2
#         do
#             q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}1_min${num_bins}_expand8@${seed}
#             save_name=qwen-2.5-${model_size}B-instruct_${q_type}
#             # no rotate
            
#             mapfile new_jobs < <(add_additional_jobs "Qwen/Qwen2.5-${model_size}B-Instruct" "qwen-2.5-${model_size}B-instruct" ${save_name})

#             for item in "${new_jobs[@]}"; do
#                 echo $item
#             done

#             jobs+=("${new_jobs[@]}")
#         done
#     done
# done


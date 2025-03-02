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

# for i in 1
# do
#     for num_bins in 0.1 0.05 0.02 0.01 0.005
#     do
#         nsamples=${nsamples_list[i]}
#         train_seqlen=${train_seqlen_list[i]}
#         val_size=${val_size_list[i]}
#         for w_bits in 3
#         do
#             for attn_weighting in input_magnitude output_magnitude org_attn_none
#             do
#                 for seed in 0 1 2
#                 do
#                     q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}_none1_min${num_bins}@${seed}
#                     save_name=llama3-8b-instruct_${q_type}
#                     # no rotate

#                     job="eval cd /playpen-ssd/ylsung/quant_repo_everything; \
#                     python fake_quant/main.py \
#                     --model meta-llama/Meta-Llama-3-8B-Instruct \
#                     --w_bits ${w_bits} --w_clip \
#                     --rotate \
#                     --seed ${seed} \
#                     --val_size ${val_size} \
#                     --min_value ${num_bins} \
#                     --max_value 1 \
#                     --add_until_fail \
#                     --module_input_weighting_yaml fake_quant/configs/input_weighting/${attn_weighting}.yaml \
#                     --nsamples ${nsamples} \
#                     --train_seqlen ${train_seqlen} \
#                     --wandb_project QuaRot \
#                     --wandb_id ylsung \
#                     --save_name ${save_name} \
#                     --lm_eval \
#                     --save_qmodel_path /playpen-ssd/ylsung/checkpoints/${save_name}.pth \
#                     --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

#                     echo $job
#                     jobs+=("$job")
#                 done
#             done
            
#         done
#     done
# done


# for i in 1
# do
#     for num_bins in 0.005
#     do
#         nsamples=${nsamples_list[i]}
#         train_seqlen=${train_seqlen_list[i]}
#         val_size=${val_size_list[i]}
#         for w_bits in 3
#         do
#             for attn_weighting in input_magnitude output_magnitude
#             do
#                 for seed in 0 1 2
#                 do
#                     q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}_none1_min${num_bins}_expand8@${seed}
#                     save_name=llama3-8b-instruct_${q_type}
#                     # no rotate

#                     job="eval cd /playpen-ssd/ylsung/quant_repo_everything; \
#                     python fake_quant/main.py \
#                     --model meta-llama/Meta-Llama-3-8B-Instruct \
#                     --w_bits ${w_bits} --w_clip \
#                     --rotate \
#                     --seed ${seed} \
#                     --val_size ${val_size} \
#                     --min_value ${num_bins} \
#                     --max_value 1 \
#                     --add_until_fail \
#                     --offload_activations \
#                     --expand_factor 8 \
#                     --module_input_weighting_yaml fake_quant/configs/input_weighting/${attn_weighting}.yaml \
#                     --nsamples ${nsamples} \
#                     --train_seqlen ${train_seqlen} \
#                     --wandb_project QuaRot \
#                     --wandb_id ylsung \
#                     --save_name ${save_name} \
#                     --lm_eval \
#                     --save_qmodel_path /playpen-ssd/ylsung/checkpoints/${save_name}.pth \
#                     --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

#                     echo $job
#                     jobs+=("$job")
#                 done
#             done
            
#         done
#     done
# done


for i in 1
do
    for num_bins in 0.005
    do
        nsamples=${nsamples_list[i]}
        train_seqlen=${train_seqlen_list[i]}
        val_size=${val_size_list[i]}
        for w_bits in 3
        do
            for attn_weighting in mindiff maxdist freqlow
            do
                for seed in 0 1 2
                do
                    q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}_none1_min${num_bins}_expand8@${seed}
                    save_name=llama3-8b-instruct_${q_type}
                    # no rotate

                    job="eval cd /playpen-ssd/ylsung/quant_repo_everything; \
                    python fake_quant/main.py \
                    --model meta-llama/Meta-Llama-3-8B-Instruct \
                    --w_bits ${w_bits} --w_clip \
                    --rotate \
                    --seed ${seed} \
                    --val_size ${val_size} \
                    --min_value ${num_bins} \
                    --max_value 1 \
                    --add_until_fail \
                    --offload_activations \
                    --expand_factor 8 \
                    --module_input_weighting_yaml fake_quant/configs/input_weighting/${attn_weighting}.yaml \
                    --nsamples ${nsamples} \
                    --train_seqlen ${train_seqlen} \
                    --wandb_project QuaRot \
                    --wandb_id ylsung \
                    --save_name ${save_name} \
                    --lm_eval \
                    --save_qmodel_path /playpen-ssd/ylsung/checkpoints/${save_name}.pth \
                    --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

                    echo $job
                    jobs+=("$job")
                done
            done
            
        done
    done
done

# i=1
# nsamples=${nsamples_list[i]}
# train_seqlen=${train_seqlen_list[i]}
# val_size=${val_size_list[i]}
# for w_bits in 3
# do
#     for attn_weighting in 0_8 0_31_32
#     do
#         for seed in 0 1 2
#         do
#             q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_masking${attn_weighting}_expand8@${seed}
#             save_name=llama3-8b-instruct_${q_type}
#             # no rotate
            
#             job="eval cd /playpen-ssd/ylsung/quant_repo_everything; \
#             python fake_quant/main.py \
#             --model meta-llama/Meta-Llama-3-8B-Instruct \
#             --w_bits ${w_bits} --w_clip \
#             --seed ${seed} \
#             --val_size ${val_size} \
#             --rotate \
#             --offload_activations \
#             --expand_factor 8 \
#             --module_input_weighting_yaml fake_quant/configs/input_weighting/masking_first.yaml \
#             --adhoc_weighting_method_type ${attn_weighting} \
#             --train_seqlen ${train_seqlen} \
#             --nsamples ${nsamples} \
#             --wandb_project QuaRot \
#             --wandb_id ylsung \
#             --save_name ${save_name} \
#             --lm_eval \
#             --save_qmodel_path /playpen-ssd/ylsung/checkpoints/${save_name}.pth \
#             --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

#             echo $job
#             jobs+=("$job")
#         done
#     done
# done


# for i in 1
# do
#     for num_bins in 0.1 0.05 0.02 0.01 0.005
#     do
#         nsamples=${nsamples_list[i]}
#         train_seqlen=${train_seqlen_list[i]}
#         val_size=${val_size_list[i]}
#         for w_bits in 3
#         do
#             for attn_weighting in input_magnitude output_magnitude
#             do
#                 for seed in 0 1 2
#                 do
#                     q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}_none1_min${num_bins}@${seed}
#                     save_name=llama3-8b-instruct_${q_type}
#                     # no rotate
                    
#                     mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name})

#                     for item in "${new_jobs[@]}"; do
#                         echo $item
#                     done

#                     jobs+=("${new_jobs[@]}")
#                 done
#             done
#         done
#     done
# done


# for i in 1
# do
#     for num_bins in 0.1
#     do
#         nsamples=${nsamples_list[i]}
#         train_seqlen=${train_seqlen_list[i]}
#         val_size=${val_size_list[i]}
#         for w_bits in 3
#         do
#             for attn_weighting in output_magnitude
#             do
#                 for seed in 1
#                 do
#                     q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}_none1_min${num_bins}@${seed}
#                     save_name=llama3-8b-instruct_${q_type}
#                     # no rotate
                    
#                     mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name})

#                     for item in "${new_jobs[@]}"; do
#                         echo $item
#                     done

#                     jobs+=("${new_jobs[@]}")
#                 done
#             done
#         done
#     done
# done

# for i in 1
# do
#     for num_bins in 0.01
#     do
#         nsamples=${nsamples_list[i]}
#         train_seqlen=${train_seqlen_list[i]}
#         val_size=${val_size_list[i]}
#         for w_bits in 3
#         do
#             for attn_weighting in output_magnitude
#             do
#                 for seed in 2
#                 do
#                     q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}_none1_min${num_bins}@${seed}
#                     save_name=llama3-8b-instruct_${q_type}
#                     # no rotate
                    
#                     mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name})

#                     for item in "${new_jobs[@]}"; do
#                         echo $item
#                     done

#                     jobs+=("${new_jobs[@]}")
#                 done
#             done
#         done
#     done
# done


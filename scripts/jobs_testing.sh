source /home/ylsung/codes/env.sh

nsamples_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
train_seqlen_list=(1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024 512 256)

# 3 to 12
# 3 6 9 12
nsamples_list=(128 256 512 1024 2048 8192 32768 1048576)
train_seqlen_list=(8192 4096 2048 1024 512 128 32 1)
val_size_list=(16 32 64 128 256 1024 4096 131072)

declare -a jobs

source scripts/additional_eval_func.sh

i=3


# for custom_attn_type in block
# do
#     for attn_length in 512 128
#     do
#         nsamples=${nsamples_list[i]}
#         train_seqlen=${train_seqlen_list[i]}
#         val_size=${val_size_list[i]}
#         attn_weighting=org_attn_none
#         num_bins=5
#         for w_bits in 3
#         do
#             for seed in 0 1 2
#             do
#                 q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${custom_attn_type}${attn_length}_${attn_weighting}${num_bins}@${seed}
#                 save_name=llama3-8b-instruct_${q_type}
#                 # no rotate
                
#                 job="eval cd /home/ylsung/codes/QuaRot/; \
#                 python fake_quant/main.py \
#                 --model meta-llama/Meta-Llama-3-8B-Instruct \
#                 --w_bits ${w_bits} --w_clip \
#                 --seed ${seed} \
#                 --rotate \
#                 --val_size ${val_size} \
#                 --nsamples ${nsamples} \
#                 --train_seqlen ${train_seqlen} \
#                 --custom_attn_type ${custom_attn_type} \
#                 --attn_length ${attn_length} \
#                 --max_value ${num_bins} \
#                 --module_input_weighting_yaml fake_quant/configs/input_weighting/${attn_weighting}3.yaml \
#                 --wandb_project QuaRot \
#                 --wandb_id ylsung \
#                 --save_name ${save_name} \
#                 --lm_eval \
#                 --save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
#                 --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

#                 echo $job
#                 jobs+=("$job")
#             done
#         done
#     done
# done


for masking in None
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    val_size=${val_size_list[i]}
    for w_bits in 3
    do
        for attn_weighting in input_dot input_magnitude output_magnitude
        do
            for seed in 0 1 2
            do
                q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}_none3@${seed}
                save_name=llama3-8b-instruct_${q_type}
                # no rotate

                job="eval cd /home/ylsung/codes/QuaRot/; \
                python fake_quant/main.py \
                --model meta-llama/Meta-Llama-3-8B-Instruct \
                --w_bits ${w_bits} --w_clip \
                --seed ${seed} \
                --val_size ${val_size} \
                --rotate \
                --layerwise_weighting \
                --module_input_weighting_yaml fake_quant/configs/input_weighting/${attn_weighting}.yaml \
                --nsamples ${nsamples} \
                --train_seqlen ${train_seqlen} \
                --wandb_project QuaRot \
                --wandb_id ylsung \
                --save_name ${save_name} \
                --lm_eval \
                --save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
                --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

                echo $job
                jobs+=("$job")
            done
        done
    done
done



# for num_bins in 10
# do
#     nsamples=${nsamples_list[i]}
#     train_seqlen=${train_seqlen_list[i]}
#     val_size=${val_size_list[i]}

#     for w_bits in 3
#     do
#         for attn_weighting in org_attn_none
#         do
#             for seed in 0 1 2
#             do
#                 q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}${num_bins}_bin@${seed}
#                 save_name=llama3-8b-instruct_${q_type}
#                 # no rotate

#                 job="eval cd /home/ylsung/codes/QuaRot/; \
#                 python fake_quant/main.py \
#                 --model meta-llama/Meta-Llama-3-8B-Instruct \
#                 --w_bits ${w_bits} --w_clip \
#                 --rotate \
#                 --seed ${seed} \
#                 --num_bins ${num_bins} \
#                 --max_value ${num_bins} \
#                 --val_size ${val_size} \
#                 --module_input_weighting_yaml fake_quant/configs/input_weighting/${attn_weighting}3.yaml \
#                 --nsamples ${nsamples} \
#                 --train_seqlen ${train_seqlen} \
#                 --wandb_project QuaRot \
#                 --wandb_id ylsung \
#                 --save_name ${save_name} \
#                 --lm_eval \
#                 --save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
#                 --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

#                 echo $job
#                 jobs+=("$job")
#             done
#         done
#     done
# done


# for num_bins in 10
# do
#     nsamples=${nsamples_list[i]}
#     train_seqlen=${train_seqlen_list[i]}
#     val_size=${val_size_list[i]}

#     for w_bits in 3
#     do
#         for attn_weighting in org_attn_none
#         do
#             for seed in 0 1 2
#             do
#                 q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}${num_bins}@${seed}
#                 save_name=llama3-8b-instruct_${q_type}
#                 # no rotate

#                 job="eval cd /home/ylsung/codes/QuaRot/; \
#                 python fake_quant/main.py \
#                 --model meta-llama/Meta-Llama-3-8B-Instruct \
#                 --w_bits ${w_bits} --w_clip \
#                 --rotate \
#                 --seed ${seed} \
#                 --max_value ${num_bins} \
#                 --val_size ${val_size} \
#                 --module_input_weighting_yaml fake_quant/configs/input_weighting/${attn_weighting}3.yaml \
#                 --nsamples ${nsamples} \
#                 --train_seqlen ${train_seqlen} \
#                 --wandb_project QuaRot \
#                 --wandb_id ylsung \
#                 --save_name ${save_name} \
#                 --lm_eval \
#                 --save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
#                 --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

#                 echo $job
#                 jobs+=("$job")
#             done
#         done
#     done
# done


# for custom_attn_type in block
# do
#     for attn_length in 512 128
#     do
#         nsamples=${nsamples_list[i]}
#         train_seqlen=${train_seqlen_list[i]}
#         val_size=${val_size_list[i]}
#         attn_weighting=org_attn_none
#         num_bins=5
#         for w_bits in 3
#         do
#             for seed in 0 1 2
#             do
#                 q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${custom_attn_type}${attn_length}_${attn_weighting}${num_bins}_bin@${seed}
#                 save_name=llama3-8b-instruct_${q_type}
#                 # no rotate
                
#                 mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name})

#                 for item in "${new_jobs[@]}"; do
#                     echo $item
#                 done

#                 jobs+=("${new_jobs[@]}")
#             done
#         done
#     done
# done

# for custom_attn_type in block
# do
#     for attn_length in 512 128
#     do
#         nsamples=${nsamples_list[i]}
#         train_seqlen=${train_seqlen_list[i]}
#         val_size=${val_size_list[i]}
#         attn_weighting=org_attn_none
#         num_bins=5
#         for w_bits in 3
#         do
#             for seed in 0 1 2
#             do
#                 q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${custom_attn_type}${attn_length}_${attn_weighting}${num_bins}@${seed}
#                 save_name=llama3-8b-instruct_${q_type}
#                 # no rotate
                
#                 mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name})

#                 for item in "${new_jobs[@]}"; do
#                     echo $item
#                 done

#                 jobs+=("${new_jobs[@]}")
#             done
#         done
#     done
# done


for masking in None
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    val_size=${val_size_list[i]}
    for w_bits in 3
    do
        for attn_weighting in input_dot input_magnitude output_magnitude
        do
            for seed in 0 1 2
            do
                q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}_none3@${seed}
                save_name=llama3-8b-instruct_${q_type}
                # no rotate
                
                mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name})

                for item in "${new_jobs[@]}"; do
                    echo $item
                done

                jobs+=("${new_jobs[@]}")
            done
        done
    done
done


# for num_bins in 10
# do
#     nsamples=${nsamples_list[i]}
#     train_seqlen=${train_seqlen_list[i]}
#     val_size=${val_size_list[i]}

#     for w_bits in 3
#     do
#         for attn_weighting in org_attn_none
#         do
#             for seed in 0 1 2
#             do
#                 q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}${num_bins}_bin@${seed}
#                 save_name=llama3-8b-instruct_${q_type}
#                 # no rotate
                
#                 mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name})

#                 for item in "${new_jobs[@]}"; do
#                     echo $item
#                 done

#                 jobs+=("${new_jobs[@]}")
#             done
#         done
#     done
# done


# for num_bins in 10
# do
#     nsamples=${nsamples_list[i]}
#     train_seqlen=${train_seqlen_list[i]}
#     val_size=${val_size_list[i]}

#     for w_bits in 3
#     do
#         for attn_weighting in org_attn_none
#         do
#             for seed in 0 1 2
#             do
#                 q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}${num_bins}@${seed}
#                 save_name=llama3-8b-instruct_${q_type}
#                 # no rotate
                
#                 mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name})

#                 for item in "${new_jobs[@]}"; do
#                     echo $item
#                 done

#                 jobs+=("${new_jobs[@]}")
#             done
#         done
#     done
# done

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

i=1

num_bins=0.005
attn_weighting=org_attn_none


nsamples=${nsamples_list[i]}
train_seqlen=${train_seqlen_list[i]}
val_size=${val_size_list[i]}


# w_bits=16
# q_type=${w_bits}bit
# save_name=mistral-7b-instruct-v0.3_${q_type}
# # no rotate

# job="eval cd /home/ylsung/codes/QuaRot/; \
# python fake_quant/main.py \
# --model mistralai/Mistral-7B-Instruct-v0.3 \
# --w_bits ${w_bits} --w_clip \
# --wandb_project QuaRot \
# --wandb_id ylsung \
# --save_name ${save_name} \
# --lm_eval \
# --rotate \
# --save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
# --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

# echo $job
# jobs+=("$job")


# w_bits=16
# q_type=${w_bits}bit
# save_name=mistral-small-instruct-2409_${q_type}
# # no rotate

# job="eval cd /home/ylsung/codes/QuaRot/; \
# python fake_quant/main.py \
# --model mistralai/Mistral-Small-Instruct-2409 \
# --w_bits ${w_bits} --w_clip \
# --wandb_project QuaRot \
# --wandb_id ylsung \
# --save_name ${save_name} \
# --lm_eval \
# --rotate \
# --save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
# --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

# echo $job
# jobs+=("$job")


# for w_bits in 3
# do
#     for seed in 0 1 2
#     do
#         q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
#         save_name=mistral-7b-instruct-v0.3_${q_type}
#         # no rotate

#         job="eval cd /home/ylsung/codes/QuaRot/; \
#         python fake_quant/main.py \
#         --model mistralai/Mistral-7B-Instruct-v0.3 \
#         --w_bits ${w_bits} --w_clip \
#         --rotate \
#         --seed ${seed} \
#         --add_until_fail \
#         --val_size ${val_size} \
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


# for w_bits in 3
# do
#     for seed in 0 1 2
#     do
#         q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
#         save_name=mistral-small-instruct-2409_${q_type}
#         # no rotate

#         job="eval cd /home/ylsung/codes/QuaRot/; \
#         python fake_quant/main.py \
#         --model mistralai/Mistral-Small-Instruct-2409 \
#         --w_bits ${w_bits} --w_clip \
#         --rotate \
#         --seed ${seed} \
#         --add_until_fail \
#         --val_size ${val_size} \
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


# w_bits=16
# q_type=${w_bits}bit
# save_name=mistral-7b-instruct-v0.3_${q_type}

# mapfile new_jobs < <(add_additional_jobs "mistralai/Mistral-7B-Instruct-v0.3" "mistral-7b-instruct-v0.3" ${save_name})

# for item in "${new_jobs[@]}"; do
#     echo $item
# done

# jobs+=("${new_jobs[@]}")

# w_bits=16
# q_type=${w_bits}bit
# save_name=mistral-small-instruct-2409_${q_type}

# mapfile new_jobs < <(add_additional_jobs "mistralai/Mistral-Small-Instruct-2409" "mistral-small-instruct-2409" ${save_name})

# for item in "${new_jobs[@]}"; do
#     echo $item
# done

# jobs+=("${new_jobs[@]}")


# for w_bits in 3
# do
#     for seed in 0 1 2
#     do
#         q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
#         save_name=mistral-7b-instruct-v0.3_${q_type}
#         # no rotate
        
#         mapfile new_jobs < <(add_additional_jobs "mistralai/Mistral-7B-Instruct-v0.3" "mistral-7b-instruct-v0.3" ${save_name})

#         for item in "${new_jobs[@]}"; do
#             echo $item
#         done

#         jobs+=("${new_jobs[@]}")
#     done
# done


for w_bits in 3
do

    for seed in 0
    do
        q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
        save_name=mistral-small-instruct-2409_${q_type}
        # no rotate
        
        mapfile new_jobs < <(add_additional_jobs "mistralai/Mistral-Small-Instruct-2409" "mistral-small-instruct-2409" ${save_name})

        for item in "${new_jobs[@]}"; do
            echo $item
        done

        jobs+=("${new_jobs[@]}")
    done
done
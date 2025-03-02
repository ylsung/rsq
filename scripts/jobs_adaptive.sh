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


for i in 3
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    val_size=${val_size_list[i]}
    for w_bits in 4
    do
        for seed in 0 1 2
        do
            q_type=${w_bits}bit_w_n${nsamples}_l${train_seqlen}_adaptive_4e-3_4e-1@${seed}
            save_name=llama3-8b-instruct_${q_type}
            # no rotate
            
            job="eval cd /home/ylsung/codes/QuaRot/; \
            python fake_quant/main.py \
            --model meta-llama/Meta-Llama-3-8B-Instruct \
            --w_bits ${w_bits} --w_clip \
            --adaptive_gptq \
            --values_to_try '[0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4]' \
            --seed ${seed} \
            --val_size ${val_size} \
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



for i in 3
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    val_size=${val_size_list[i]}
    for w_bits in 4
    do
        for seed in 0 1 2
        do
            q_type=${w_bits}bit_w_n${nsamples}_l${train_seqlen}_adaptive_4e-3_4e-1@${seed}
            save_name=llama3-8b-instruct_${q_type}
            # no rotate
            
            mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name} False)

            for item in "${new_jobs[@]}"; do
                echo $item
            done

            jobs+=("${new_jobs[@]}")

        done
    done
done


# for i in 3
# do
#     nsamples=${nsamples_list[i]}
#     train_seqlen=${train_seqlen_list[i]}
#     val_size=${val_size_list[i]}
#     for w_bits in 4
#     do
#         for seed in 0 1 2
#         do
#             q_type=${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
#             save_name=llama3-8b-instruct_${q_type}
#             # no rotate
            
#             job="eval cd /home/ylsung/codes/QuaRot/; \
#             python fake_quant/main.py \
#             --model meta-llama/Meta-Llama-3-8B-Instruct \
#             --w_bits ${w_bits} --w_clip \
#             --seed ${seed} \
#             --val_size ${val_size} \
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



# for i in 3
# do
#     nsamples=${nsamples_list[i]}
#     train_seqlen=${train_seqlen_list[i]}
#     val_size=${val_size_list[i]}
#     for w_bits in 4
#     do
#         for seed in 0 1 2
#         do
#             q_type=${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
#             save_name=llama3-8b-instruct_${q_type}
#             # no rotate
            
#             mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name} False)

#             for item in "${new_jobs[@]}"; do
#                 echo $item
#             done

#             jobs+=("${new_jobs[@]}")

#         done
#     done
# done

declare -a jobs

attn_weighting=attncon
nsamples=256
train_seqlen=4096
val_size=0
save_name_prefix=llama3-8b-instruct
model_name=meta-llama/Meta-Llama-3-8B-Instruct

# load the script to evaluate other tasks
source scripts/additional_short_eval.sh
# get the env variable for CODEPATH and CHECKPOINT_PATH
source scripts/env.sh

# min_value search over 0.1 0.05 0.02 0.01 0.005
# for min_value in 0.005
# do
#     for w_bits in 3
#     do
#         for seed in 0 1 2
#         do
#             save_name=${save_name_prefix}_rsq_${w_bits}bit_n${nsamples}_l${train_seqlen}_${attn_weighting}_min${min_value}@${seed}

#             job="eval cd ${CODEPATH}; \
#             python fake_quant/main.py \
#             --model ${model_name} \
#             --rotate \
#             --w_bits ${w_bits} --w_clip \
#             --seed ${seed} \
#             --min_value ${min_value} \
#             --max_value 1 \
#             --add_until_fail \
#             --val_size ${val_size} \
#             --module_input_weighting_yaml fake_quant/configs/input_weighting/${attn_weighting}.yaml \
#             --nsamples ${nsamples} \
#             --train_seqlen ${train_seqlen} \
#             --save_name ${save_name} \
#             --lm_eval \
#             --save_qmodel_path ${CHECKPOINT_PATH}/${save_name}.pth \
#             --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

#             echo $job
#             jobs+=("$job")
#         done
#     done
# done

for min_value in 0.005
do
    for w_bits in 3
    do
        for seed in 0
        do
            save_name=${save_name_prefix}_rsq_${w_bits}bit_n${nsamples}_l${train_seqlen}_${attn_weighting}_min${min_value}@${seed}
            
            mapfile new_jobs < <(add_additional_jobs "${model_name}" "${save_name_prefix}" ${save_name} True)

            for item in "${new_jobs[@]}"; do
                echo $item
            done

            jobs+=("${new_jobs[@]}")
        done
    done
done

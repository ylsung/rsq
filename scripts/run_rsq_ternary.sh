declare -a jobs

# scaling_strategy=attncon
# nsamples=256
# train_seqlen=4096

# ### For LLaMA
# save_name_prefix=llama3-8b-instruct
# model_name=meta-llama/Meta-Llama-3-8B-Instruct

# # ### For Mistral
# # model_name=mistralai/Mistral-Nemo-Instruct-2407
# # model_store_name=mistral-nemo-instruct-2407

# # ### For Qwen
# # model_size=7
# # save_name_prefix=qwen-2.5-${model_size}B-instruct
# # model_name=Qwen/Qwen2.5-${model_size}B-Instruct

# source scripts/additional_short_eval.sh
# source scripts/env.sh

# w_bits=2
# method_name=rsq_ternary
# w_quant_scheme=ternary
# min_value=0.005

# for seed in 0 1 2
# do
#     save_name=${save_name_prefix}_${method_name}_n${nsamples}_l${train_seqlen}_${scaling_strategy}_min${min_value}@${seed}

#     job="eval cd ${CODEPATH}; \
#     python fake_quant/main.py \
#     --model ${model_name} \
#     --rotate \
#     --w_bits ${w_bits} --w_clip \
#     --w_quant_scheme ${w_quant_scheme} \
#     --seed ${seed} \
#     --min_value ${min_value} \
#     --max_value 1 \
#     --add_until_fail \
#     --module_input_weighting_yaml fake_quant/configs/input_weighting/${scaling_strategy}.yaml \
#     --nsamples ${nsamples} \
#     --train_seqlen ${train_seqlen} \
#     --save_name ${save_name} \
#     --lm_eval \
#     --save_qmodel_path ${CHECKPOINT_PATH}/${save_name}.pth \
#     --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

#     echo $job
#     jobs+=("$job")
# done

# for seed in 0 1 2
# do
#     save_name=${save_name_prefix}_${method_name}_n${nsamples}_l${train_seqlen}_${scaling_strategy}_min${min_value}@${seed}

#     mapfile new_jobs < <(add_additional_jobs "${model_name}" "${save_name_prefix}" ${save_name} True)

#     for item in "${new_jobs[@]}"; do
#         echo $item
#     done

#     jobs+=("${new_jobs[@]}")
# done


# w_bits=2
# method_name=quarot_ternary
# w_quant_scheme=ternary


# ### For LLaMA 
# save_name_prefix=llama3-8b-instruct
# model_name=meta-llama/Meta-Llama-3-8B-Instruct

# # ### For Mistral
# # model_name=mistralai/Mistral-Nemo-Instruct-2407
# # model_store_name=mistral-nemo-instruct-2407

# # ### For Qwen
# # model_size=7
# # save_name_prefix=qwen-2.5-${model_size}B-instruct
# # model_name=Qwen/Qwen2.5-${model_size}B-Instruct

# # load the script to evaluate other tasks
# source scripts/additional_short_eval.sh
# # get the env variable for CODEPATH and CHECKPOINT_PATH
# source scripts/env.sh


# for seed in 0 1 2
# do
#     save_name=${save_name_prefix}_${method_name}_${w_bits}bit_n${nsamples}_l${train_seqlen}@${seed}

#     job="eval cd ${CODEPATH}; \
#     python fake_quant/main.py \
#     --model ${model_name} \
#     --rotate \
#     --w_bits ${w_bits} --w_clip \
#     --w_quant_scheme ${w_quant_scheme} \
#     --seed ${seed} \
#     --add_until_fail \
#     --nsamples ${nsamples} \
#     --train_seqlen ${train_seqlen} \
#     --save_name ${save_name} \
#     --lm_eval \
#     --save_qmodel_path ${CHECKPOINT_PATH}/${save_name}.pth \
#     --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

#     echo $job
#     jobs+=("$job")
# done

# # Evaluation for Wiki-len512, Wiki-len8192, MMLU, GSM8K, TruthfulQA
# for seed in 0 1 2
# do
#     save_name=${save_name_prefix}_${method_name}_${w_bits}bit_n${nsamples}_l${train_seqlen}@${seed}
    
#     mapfile new_jobs < <(add_additional_jobs "${model_name}" "${save_name_prefix}" ${save_name} True)

#     for item in "${new_jobs[@]}"; do
#         echo $item
#     done

#     jobs+=("${new_jobs[@]}")
# done


# qwen 72B


scaling_strategy=attncon
nsamples=64
train_seqlen=4096

### For LLaMA
# save_name_prefix=llama3-8b-instruct
# model_name=meta-llama/Meta-Llama-3-8B-Instruct

# ### For Mistral
# model_name=mistralai/Mistral-Nemo-Instruct-2407
# model_store_name=mistral-nemo-instruct-2407

# ### For Qwen
model_size=72
save_name_prefix=qwen-2.5-${model_size}B-instruct
model_name=Qwen/Qwen2.5-${model_size}B-Instruct

source scripts/additional_short_eval.sh
source scripts/env.sh

w_bits=2
method_name=rsq
min_value=0.005
# w_quant_scheme=ternary

for seed in 0
do
    save_name=${save_name_prefix}_${method_name}_n${nsamples}_l${train_seqlen}_${scaling_strategy}_min${min_value}@${seed}

    job="eval cd ${CODEPATH}; \
    python fake_quant/main.py \
    --model ${model_name} \
    --rotate \
    --w_bits ${w_bits} --w_clip \
    --seed ${seed} \
    --min_value ${min_value} \
    --max_value 1 \
    --add_until_fail \
    --module_input_weighting_yaml fake_quant/configs/input_weighting/${scaling_strategy}.yaml \
    --nsamples ${nsamples} \
    --train_seqlen ${train_seqlen} \
    --save_name ${save_name} \
    --lm_eval \
    --save_qmodel_path ${CHECKPOINT_PATH}/${save_name}.pth \
    --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

    echo $job
    jobs+=("$job")
done

# for seed in 0
# do
#     save_name=${save_name_prefix}_${method_name}_n${nsamples}_l${train_seqlen}_${scaling_strategy}_min${min_value}@${seed}

#     mapfile new_jobs < <(add_additional_jobs "${model_name}" "${save_name_prefix}" ${save_name} True)

#     for item in "${new_jobs[@]}"; do
#         echo $item
#     done

#     jobs+=("${new_jobs[@]}")
# done


w_bits=2
method_name=quarot


### For LLaMA 
# save_name_prefix=llama3-8b-instruct
# model_name=meta-llama/Meta-Llama-3-8B-Instruct

# ### For Mistral
# model_name=mistralai/Mistral-Nemo-Instruct-2407
# model_store_name=mistral-nemo-instruct-2407

# ### For Qwen
model_size=72
save_name_prefix=qwen-2.5-${model_size}B-instruct
model_name=Qwen/Qwen2.5-${model_size}B-Instruct

# load the script to evaluate other tasks
source scripts/additional_short_eval.sh
# get the env variable for CODEPATH and CHECKPOINT_PATH
source scripts/env.sh


for seed in 0
do
    save_name=${save_name_prefix}_${method_name}_${w_bits}bit_n${nsamples}_l${train_seqlen}@${seed}

    job="eval cd ${CODEPATH}; \
    python fake_quant/main.py \
    --model ${model_name} \
    --rotate \
    --w_bits ${w_bits} --w_clip \
    --seed ${seed} \
    --add_until_fail \
    --nsamples ${nsamples} \
    --train_seqlen ${train_seqlen} \
    --save_name ${save_name} \
    --lm_eval \
    --save_qmodel_path ${CHECKPOINT_PATH}/${save_name}.pth \
    --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

    echo $job
    jobs+=("$job")
done

# # Evaluation for Wiki-len512, Wiki-len8192, MMLU, GSM8K, TruthfulQA
# for seed in 0
# do
#     save_name=${save_name_prefix}_${method_name}_${w_bits}bit_n${nsamples}_l${train_seqlen}@${seed}
    
#     mapfile new_jobs < <(add_additional_jobs "${model_name}" "${save_name_prefix}" ${save_name} True)

#     for item in "${new_jobs[@]}"; do
#         echo $item
#     done

#     jobs+=("${new_jobs[@]}")
# done


# qwen 72B ternary


scaling_strategy=attncon
nsamples=64
train_seqlen=4096

### For LLaMA
# save_name_prefix=llama3-8b-instruct
# model_name=meta-llama/Meta-Llama-3-8B-Instruct

# ### For Mistral
# model_name=mistralai/Mistral-Nemo-Instruct-2407
# model_store_name=mistral-nemo-instruct-2407

# ### For Qwen
model_size=72
save_name_prefix=qwen-2.5-${model_size}B-instruct
model_name=Qwen/Qwen2.5-${model_size}B-Instruct

source scripts/additional_short_eval.sh
source scripts/env.sh

w_bits=2
method_name=rsq_ternary
min_value=0.005
w_quant_scheme=ternary

for seed in 0
do
    save_name=${save_name_prefix}_${method_name}_n${nsamples}_l${train_seqlen}_${scaling_strategy}_min${min_value}@${seed}

    job="eval cd ${CODEPATH}; \
    python fake_quant/main.py \
    --model ${model_name} \
    --rotate \
    --w_bits ${w_bits} --w_clip \
    --w_quant_scheme ${w_quant_scheme} \
    --seed ${seed} \
    --min_value ${min_value} \
    --max_value 1 \
    --add_until_fail \
    --module_input_weighting_yaml fake_quant/configs/input_weighting/${scaling_strategy}.yaml \
    --nsamples ${nsamples} \
    --train_seqlen ${train_seqlen} \
    --save_name ${save_name} \
    --lm_eval \
    --save_qmodel_path ${CHECKPOINT_PATH}/${save_name}.pth \
    --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

    echo $job
    jobs+=("$job")
done

# for seed in 0
# do
#     save_name=${save_name_prefix}_${method_name}_n${nsamples}_l${train_seqlen}_${scaling_strategy}_min${min_value}@${seed}

#     mapfile new_jobs < <(add_additional_jobs "${model_name}" "${save_name_prefix}" ${save_name} True)

#     for item in "${new_jobs[@]}"; do
#         echo $item
#     done

#     jobs+=("${new_jobs[@]}")
# done


w_bits=2
method_name=quarot_ternary


### For LLaMA 
# save_name_prefix=llama3-8b-instruct
# model_name=meta-llama/Meta-Llama-3-8B-Instruct

# ### For Mistral
# model_name=mistralai/Mistral-Nemo-Instruct-2407
# model_store_name=mistral-nemo-instruct-2407

# ### For Qwen
model_size=72
save_name_prefix=qwen-2.5-${model_size}B-instruct
model_name=Qwen/Qwen2.5-${model_size}B-Instruct

# load the script to evaluate other tasks
source scripts/additional_short_eval.sh
# get the env variable for CODEPATH and CHECKPOINT_PATH
source scripts/env.sh


for seed in 0
do
    save_name=${save_name_prefix}_${method_name}_${w_bits}bit_n${nsamples}_l${train_seqlen}@${seed}

    job="eval cd ${CODEPATH}; \
    python fake_quant/main.py \
    --model ${model_name} \
    --rotate \
    --w_bits ${w_bits} --w_clip \
    --w_quant_scheme ${w_quant_scheme} \
    --seed ${seed} \
    --add_until_fail \
    --nsamples ${nsamples} \
    --train_seqlen ${train_seqlen} \
    --save_name ${save_name} \
    --lm_eval \
    --save_qmodel_path ${CHECKPOINT_PATH}/${save_name}.pth \
    --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

    echo $job
    jobs+=("$job")
done

# # Evaluation for Wiki-len512, Wiki-len8192, MMLU, GSM8K, TruthfulQA
# for seed in 0
# do
#     save_name=${save_name_prefix}_${method_name}_${w_bits}bit_n${nsamples}_l${train_seqlen}@${seed}
    
#     mapfile new_jobs < <(add_additional_jobs "${model_name}" "${save_name_prefix}" ${save_name} True)

#     for item in "${new_jobs[@]}"; do
#         echo $item
#     done

#     jobs+=("${new_jobs[@]}")
# done

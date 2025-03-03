declare -a jobs

scaling_strategy=attncon
nsamples=256
train_seqlen=4096
method_name=rsq


### For LLaMA 
save_name_prefix=llama3-8b-instruct
model_name=meta-llama/Meta-Llama-3-8B-Instruct

# ### For Mistral
# model_name=mistralai/Mistral-Nemo-Instruct-2407
# model_store_name=mistral-nemo-instruct-2407

# ### For Qwen
# model_size=7
# save_name_prefix=qwen-2.5-${model_size}B-instruct
# model_name=Qwen/Qwen2.5-${model_size}B-Instruct

# load the script to evaluate other tasks
source scripts/additional_short_eval.sh
# get the env variable for CODEPATH and CHECKPOINT_PATH
source scripts/env.sh

# scaling_strategy can be attncon actdiff actnorm tokenfreq tokensim
# min_value search over 0.1 0.05 0.02 0.01 0.005
for min_value in 0.005
do
    for seed in 0 1 2
    do
        save_name=${save_name_prefix}_${method_name}_e8p_n${nsamples}_l${train_seqlen}_${scaling_strategy}_min${min_value}@${seed}

        job="eval cd ${CODEPATH}; \
        python fake_quant/main.py \
        --model ${model_name} \
        --rotate \
        --w_bits 2 --w_clip \
        --seed ${seed} \
        --min_value ${min_value} \
        --max_value 1 \
        --add_until_fail \
        --e8p \
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
done

# Evaluation for Wiki-len512, Wiki-len8192, MMLU, GSM8K, TruthfulQA
for min_value in 0.005
do
    for seed in 0 1 2
    do
        save_name=${save_name_prefix}_${method_name}_e8p_n${nsamples}_l${train_seqlen}_${scaling_strategy}_min${min_value}@${seed}
        
        mapfile new_jobs < <(add_additional_jobs "${model_name}" "${save_name_prefix}" ${save_name} True)

        for item in "${new_jobs[@]}"; do
            echo $item
        done

        jobs+=("${new_jobs[@]}")
    done
done

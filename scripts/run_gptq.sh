declare -a jobs

nsamples=256
train_seqlen=4096
method_name=gptq


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


for w_bits in 3
do
    for seed in 0 1 2
    do
        save_name=${save_name_prefix}_${method_name}_${w_bits}bit_n${nsamples}_l${train_seqlen}@${seed}

        job="eval cd ${CODEPATH}; \
        python fake_quant/main.py \
        --model ${model_name} \
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
done

# Evaluation for Wiki-len512, Wiki-len8192, MMLU, GSM8K, TruthfulQA
for w_bits in 3
do
    for seed in 0 1 2
    do
        save_name=${save_name_prefix}_${method_name}_${w_bits}bit_n${nsamples}_l${train_seqlen}@${seed}
        
        mapfile new_jobs < <(add_additional_jobs "${model_name}" "${save_name_prefix}" ${save_name} False)

        for item in "${new_jobs[@]}"; do
            echo $item
        done

        jobs+=("${new_jobs[@]}")
    done
done

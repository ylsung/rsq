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

w_bits=16
save_name=${save_name_prefix}_${w_bits}bit

job="eval cd ${CODEPATH}; \
python fake_quant/main.py \
--model ${model_name} \
--rotate \
--w_bits ${w_bits} --w_clip \
--save_name ${save_name} \
--lm_eval \
--save_qmodel_path ${CHECKPOINT_PATH}/${save_name}.pth \
--tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

echo $job
jobs+=("$job")

# Evaluation for Wiki-len512, Wiki-len8192, MMLU, GSM8K, TruthfulQA
save_name=${save_name_prefix}_${w_bits}bit

mapfile new_jobs < <(add_additional_jobs "${model_name}" "${save_name_prefix}" ${save_name} True)

for item in "${new_jobs[@]}"; do
    echo $item
done

jobs+=("${new_jobs[@]}")

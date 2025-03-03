# load the script to evaluate other tasks
source scripts/additional_long_eval.sh
# get the env variable for CODEPATH and CHECKPOINT_PATH
source scripts/env.sh

##### Scripts for long-context tasks are only test for LLaMA
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

# checkpoints want to be evaluated, just use the stem of the checkpoint path, 
# exluding the parent folder because it is specify in CHECKPOINT_PATH
# for example if the path of a checkpoint is .../.../AAA.pth, then use AAA here
checkpoints_to_eval=(
    AAA
    BBB
    CCC
)

# set the argument to True if the checkpoints are rotated: QuaRot and RSQ
# set it to false if not rotated: GPTQ

for checkpoint in ${checkpoints_to_eval[@]}
do
    mapfile new_jobs < <(add_long_eval_jobs "${model_name}" "${save_name_prefix}" ${checkpoint} True)

    for item in "${new_jobs[@]}"; do
        echo $item
    done

    jobs+=("${new_jobs[@]}")
done
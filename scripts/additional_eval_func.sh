add_additional_jobs() {
    local full_model_name=$1  # First input string
    local short_model_name=$2  # Second input string
    local save_name=$3  # Third input string
    local rotate=${4:-True}  # Fourth input string

    if [[ $rotate == "True" ]]; then
        local rotate="--rotate"
    else
        local rotate=""
    fi

    local jobs=()

    # Wiki tasks
    for val_seqlen in 512 8192
    do
        # no rotate
        job="eval cd /playpen-ssd/ylsung/quant_repo_everything/; \
        python fake_quant/main.py \
        --model ${full_model_name} \
        --w_bits 16 --w_clip \
        ${rotate} \
        --val_seqlen ${val_seqlen} \
        --wandb_project QuaRot \
        --wandb_id ylsung \
        --bsz 2 \
        --save_name ${save_name}_wiki${val_seqlen} \
        --load_qmodel_path /playpen-ssd2/ylsung/checkpoints/${save_name}.pth"

        # echo $job
        jobs+=("$job")
    done

    job="eval cd /playpen-ssd/ylsung/quant_repo_everything/; \
    python fake_quant/main.py \
    --model ${full_model_name} \
    --w_bits 16 --w_clip \
    ${rotate} \
    --wandb_project QuaRot \
    --wandb_id ylsung \
    --save_name ${save_name}_truthfulqa \
    --lm_eval \
    --lm_eval_batch_size 8 \
    --load_qmodel_path /playpen-ssd2/ylsung/checkpoints/${save_name}.pth \
    --tasks truthfulqa_mc2"

    jobs+=("$job")

    job="eval cd /playpen-ssd/ylsung/quant_repo_everything/; \
    python fake_quant/main.py \
    --model ${full_model_name} \
    --w_bits 16 --w_clip \
    ${rotate} \
    --wandb_project QuaRot \
    --wandb_id ylsung \
    --save_name ${save_name}_mmlu \
    --lm_eval \
    --lm_eval_batch_size 1 \
    --load_qmodel_path /playpen-ssd2/ylsung/checkpoints/${save_name}.pth \
    --tasks mmlu \
    --num_fewshot 5"

    jobs+=("$job")


    job="eval cd /playpen-ssd/ylsung/quant_repo_everything/; \
    python fake_quant/main.py \
    --model ${full_model_name} \
    --w_bits 16 --w_clip \
    ${rotate} \
    --wandb_project QuaRot \
    --wandb_id ylsung \
    --save_name ${save_name}_gsm8k \
    --lm_eval \
    --load_qmodel_path /playpen-ssd2/ylsung/checkpoints/${save_name}.pth \
    --tasks gsm8k_cot_llama \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --num_fewshot 8"

    jobs+=("$job")

    for job in "${jobs[@]}"; do
        echo "$job"
    done

    # Print the modified list
    # echo "${jobs[@]}"
}
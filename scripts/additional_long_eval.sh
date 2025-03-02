add_long_eval_jobs() {
    local full_model_name=$1  # First input string
    local short_model_name=$2  # Second input string
    local save_name=$3  # Third input string
    local rotate=${4:-True}  # Fourth input string

    if [[ $rotate == "True" ]]; then
        local rotate="--rotate"
        local rotate_checkpoint="--rotated_quantized_checkpoint"
    else
        local rotate=""
        local rotate_checkpoint="--quantized_checkpoint"
    fi

    if echo "$full_model_name" | grep -q "llama"; then
        local le_eval_file="Baselines/llama3-instruct-test-new.py"
    else
        local le_eval_file="Baselines/general_instruct-test-new.py"
    fi

    local jobs=()

    # long eval
    job="eval cd ${CODEPATH}/qllm-eval/qllm_eval/evaluation/q_long/; \
    python main_longeval_quarot.py \
    --model-name-or-path ${full_model_name} --use_flash_attn \
    --task lines --test_dir new_cases \
    --sub_task 300 460 620 \
    --prefix ${save_name}_ \
    ${rotate_checkpoint} ${CHECKPOINT_PATH}/${save_name}.pth"
    
    jobs+=("$job")
    
    # LE_EVAL
    max_length=8k
    for task_name in tpo quality coursera sci_fi gsm100 codeU topic_retrieval_longchat
    do
        # inference and evaluation
        job="eval cd ${CODEPATH}/LEval; \
        python ${le_eval_file} \
        --metric exam_eval \
        --max_length ${max_length} \
        --gpu 0 \
        --task_name ${task_name} \
        --save_name ${save_name} \
        --model_name ${full_model_name} \
        ${rotate_checkpoint} ${CHECKPOINT_PATH}/${save_name}.pth"

        jobs+=("$job")

    done

    # Tacred
    for round in 1 2
    do
        job="eval cd ${CODEPATH}/LongICLBench; \
        python my_tacred_infer_chat_new.py \
        --model ${full_model_name} \
        --round ${round} \
        --test_number 500 \
        --save_name tacred_round_result/${save_name}_${round}.json \
        ${rotate_checkpoint} ${CHECKPOINT_PATH}/${save_name}.pth"

        jobs+=("$job")
    done
    
    # Banking77
    for round in 2 3
    do
        job="eval cd ${CODEPATH}/LongICLBench; \
        python my_banking77_infer_chat_new.py \
        --model ${full_model_name} \
        --round ${round} \
        --test_number 500 \
        --save_name bank77_round_result/${save_name}_${round}.json \
        ${rotate_checkpoint} ${CHECKPOINT_PATH}/${save_name}.pth"

        jobs+=("$job")
    done

    # long code arena
    job="eval cd ${CODEPATH}/lca-baselines/library_based_code_generation; \
    python -m src.evaluation.evaluate_new \
    --model ${full_model_name}  \
    --save_name ${save_name} \
    ${rotate_checkpoint} ${CHECKPOINT_PATH}/${save_name}.pth"

    jobs+=("$job")

    # lost in the middle
    for pos in 0 14 29
    do
        job="eval cd ${CODEPATH}/qllm-eval/qllm_eval/evaluation/q_long; \
        python main_litm_new.py \
        --model_name ${full_model_name} --use_flash_attn \
        --input_path lost-in-the-middle/qa_data/30_total_documents/nq-open-30_total_documents_gold_at_${pos}.jsonl.gz \
        --max_new_tokens 100 --output_path ./litm/${save_name}/30_total_documents_gold_at_${pos}.txt \
        ${rotate_checkpoint} ${CHECKPOINT_PATH}/${save_name}.pth"
        
        jobs+=("$job")
    done

    for job in "${jobs[@]}"; do
        echo "$job"
    done

}

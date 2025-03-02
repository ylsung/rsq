rotate="--rotate"
rotate_checkpoint="--rotated_quantized_checkpoint"
le_eval_file="Baselines/llama3-instruct-test-new.py"

declare -a jobs

max_length=8k
save_name="llama3-8b-instruct_quarot_3bit_w_n512_l2048_org_attn_none1_min0.01_expand8@0"
full_model_name=meta-llama/Meta-Llama-3-8B-Instruct

for task_name in tpo
do
    # inference and evaluation
    job="eval cd /home/ylsung/codes/QuaRot/LEval; \
    python ${le_eval_file} \
    --metric exam_eval \
    --max_length ${max_length} \
    --gpu 0 \
    --task_name ${task_name} \
    --save_name ${save_name} \
    --model_name ${full_model_name} \
    ${rotate_checkpoint} /nas-hdd/ylsung/checkpoints/${save_name}.pth"

    jobs+=("$job")

    # # evaluation
    # job="eval cd /home/ylsung/codes/QuaRot/LEval; \
    # python Evaluation/auto_eval.py \
    # --pred_file Predictions/exam_eval/${save_name}-${max_length}/${task_name}.pred.jsonl"

    # jobs+=("$job")
done
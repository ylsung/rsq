source /home/ylsung/codes/env.sh

nsamples_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
train_seqlen_list=(1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024 512 256)

# 3 to 12
# 3 6 9 12
nsamples_list=(16 64 256 1024 4096)
train_seqlen_list=(65536 16384 4096 1024 256)

declare -a jobs


w_bits=16
q_type=${w_bits}bit
save_name=gemma-2-9b-it_${q_type}
# no rotate

job="eval cd /home/ylsung/codes/QuaRot/; \
python fake_quant/main.py \
--model google/gemma-2-9b-it \
--w_bits ${w_bits} --w_clip \
--wandb_project QuaRot \
--wandb_id ylsung \
--save_name ${save_name} \
--lm_eval \
--tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

echo $job
jobs+=("$job")
 

# # long context testing
# job="eval cd /home/ylsung/codes/QuaRot/qllm-eval/qllm_eval/evaluation/q_long/; \
# python main_longeval_quarot.py \
# --model-name-or-path google/gemma-2-9b-it --use_flash_attn \
# --task lines --test_dir new_cases \
# --sub_task 300 460 620 620_s0_e155 620_s156_e310 620_s311_e465 620_s466_e620 \
# --prefix ${q_type}_ \
# --quantized_checkpoint /nas-hdd/ylsung/checkpoints/gemma-2-9b-it_${q_type}.pth"

# echo $job
# jobs+=("$job")


# max_length=8k
# job="eval cd /home/ylsung/codes/QuaRot/LEval; \
# python Baselines/gemma_instruct-test.py \
# --metric exam_eval \
# --max_length ${max_length} \
# --gpu 0 \
# --postfix _${q_type} --task_name gsm100"
# # --quantized_checkpoint /nas-hdd/ylsung/checkpoints/gemma-2-9b-it_${q_type}.pth"

# echo $job
# jobs+=("$job")


# max_length=8k
# folder=Predictions/exam_eval/gemma-2-9b-instruct-${max_length}_${q_type}
# for task in gsm100
# do
#     job="eval cd /home/ylsung/codes/QuaRot/LEval; \
#     python Evaluation/auto_eval.py \
#     --pred_file ${folder}/${task}.pred.jsonl"

#     echo $job
#     jobs+=("$job")
# done

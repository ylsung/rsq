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
save_name=mistral-nemo-instruct_${q_type}
# no rotate

job="eval cd /home/ylsung/codes/QuaRot/; \
python fake_quant/main.py \
--model mistralai/Mistral-Nemo-Instruct-2407 \
--w_bits 4 --w_clip \
--wandb_project QuaRot \
--wandb_id ylsung \
--save_name ${save_name} \
--lm_eval \
--compute_next_attn_loss \
--optimizer_yaml fake_quant/configs/optimizer/attn1e-2_gd_scale_lr1e-5.yaml \
--val_size 128 \
--seed 0 \
--nsamples 1024 \
--train_seqlen 1024 \
--save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
--tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

echo $job
jobs+=("$job")


# long context testing
# job="eval cd /home/ylsung/codes/QuaRot/qllm-eval/qllm_eval/evaluation/q_long/; \
# python main_longeval_quarot.py \
# --model-name-or-path Qwen/Qwen2.5-7B-Instruct --use_flash_attn \
# --task lines --test_dir new_cases \
# --sub_task 300 460 620 620_s0_e155 620_s156_e310 620_s311_e465 620_s466_e620 \
# --prefix ${q_type}_ \
# --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/mistral-nemo-instruct_${q_type}.pth"

# echo $job
# jobs+=("$job")


# max_length=8k
# job="eval cd /home/ylsung/codes/QuaRot/LEval; \
# python Baselines/general_instruct-test.py \
# --metric exam_eval \
# --max_length ${max_length} \
# --gpu 0 \
# --model_path mistralai/Mistral-Nemo-Instruct-2407 \
# --postfix _${q_type} \
# --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/mistral-nemo-instruct_${q_type}.pth"

# echo $job
# jobs+=("$job")


# max_length=8k
# folder=Predictions/exam_eval/mistral-nemo-instruct-${max_length}_${q_type}
# for task in tpo quality coursera sci_fi gsm100 codeU topic_retrieval_longchat
# do
#     job="eval cd /home/ylsung/codes/QuaRot/LEval; \
#     python Evaluation/auto_eval.py \
#     --pred_file ${folder}/${task}.pred.jsonl"

#     echo $job
#     jobs+=("$job")
# done

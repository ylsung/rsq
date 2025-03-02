source /home/ylsung/codes/env.sh

nsamples_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
train_seqlen_list=(1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024 512 256)

# 3 to 12
# 3 6 9 12
nsamples_list=(16 64 256 1024 4096)
train_seqlen_list=(65536 16384 4096 1024 256)

declare -a jobs


# for config in attn3e-3_gd_scale_lr1e-5
# do
#     for i in 3
#     do
#         nsamples=${nsamples_list[i]}
#         train_seqlen=${train_seqlen_list[i]}
#         for w_bits in 4
#         do
#             for seed in 2
#             do
#                 q_type_postfix=next_${config}
#                 q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${q_type_postfix}@${seed}
#                 # long context testing
#                 job="eval cd /home/ylsung/codes/QuaRot/qllm-eval/qllm_eval/evaluation/q_long/; \
#                 python main_longeval_quarot.py \
#                 --model-name-or-path meta-llama/Meta-Llama-3-8B-Instruct --use_flash_attn \
#                 --task lines --test_dir new_cases \
#                 --sub_task 300 460 620 620_s0_e155 620_s156_e310 620_s311_e465 620_s466_e620 \
#                 --prefix ${q_type}_ \
#                 --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/llama3-8b-instruct_${q_type}.pth"

#                 echo $job
#                 jobs+=("$job")
#             done
#         done
#     done
# done

# for config in attn3e-3_gd_scale_lr1e-5
# do
#     for i in 3
#     do
#         nsamples=${nsamples_list[i]}
#         train_seqlen=${train_seqlen_list[i]}
#         for w_bits in 4
#         do
#             for seed in 2
#             do
#                 q_type_postfix=next_${config}
#                 max_length=8k
#                 q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${q_type_postfix}@${seed}
#                 job="eval cd /home/ylsung/codes/QuaRot/LEval; \
#                 python Baselines/llama3-instruct-test.py \
#                 --metric exam_eval \
#                 --max_length ${max_length} \
#                 --gpu 0 \
#                 --postfix _${q_type} \
#                 --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/llama3-8b-instruct_${q_type}.pth"

#                 echo $job
#                 jobs+=("$job")
#             done
#         done
#     done
# done

for config in attn3e-3_gd_scale_lr1e-5
do
    for i in 3
    do
        nsamples=${nsamples_list[i]}
        train_seqlen=${train_seqlen_list[i]}
        for w_bits in 4
        do
            for seed in 2
            do
                q_type_postfix=next_${config}
                max_length=8k
                q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${q_type_postfix}@${seed}
                folder=Predictions/exam_eval/llama3-8B-instruct-${max_length}_${q_type}
                for task in tpo quality coursera sci_fi gsm100 codeU topic_retrieval_longchat
                do
                    job="eval cd /home/ylsung/codes/QuaRot/LEval; \
                    python Evaluation/auto_eval.py \
                    --pred_file ${folder}/${task}.pred.jsonl"

                    echo $job
                    jobs+=("$job")
                done
            done
        done
    done
done

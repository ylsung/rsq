source /home/ylsung/codes/env.sh

nsamples_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
train_seqlen_list=(1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024 512 256)

# 3 to 12
# 3 6 9 12
nsamples_list=(128 256 512 1024 2048 8192 32768)
train_seqlen_list=(8192 4096 2048 1024 512 128 32)
val_size_list=(16 32 64 128 256 1024 4096)

declare -a jobs

for i in 2 4
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    val_size=${val_size_list[i]}
    for w_bits in 3
    do
        for seed in 0 1 2
        do
            q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
            save_name=mistral-nemo-instruct-2407_${q_type}
            # no rotate
            
            job="eval cd /home/ylsung/codes/QuaRot/; \
            python fake_quant/main.py \
            --model mistralai/Mistral-Nemo-Instruct-2407 \
            --w_bits ${w_bits} --w_clip \
            --rotate \
            --seed ${seed} \
            --val_size ${val_size} \
            --nsamples ${nsamples} \
            --train_seqlen ${train_seqlen} \
            --wandb_project QuaRot \
            --wandb_id ylsung \
            --save_name ${save_name} \
            --lm_eval \
            --save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
            --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

            echo $job
            jobs+=("$job")
        done
    done
done

for i in 2 4
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    for w_bits in 3
    do
        for seed in 0 1 2
        do
            q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
            # long context testing
            job="eval cd /home/ylsung/codes/QuaRot/qllm-eval/qllm_eval/evaluation/q_long/; \
            python main_longeval_quarot.py \
            --model-name-or-path mistralai/Mistral-Nemo-Instruct-2407 --use_flash_attn \
            --task lines --test_dir new_cases \
            --sub_task 300 460 620 620_s0_e155 620_s156_e310 620_s311_e465 620_s466_e620 \
            --prefix ${q_type}_ \
            --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/mistral-nemo-instruct-2407_${q_type}.pth"

            echo $job
            jobs+=("$job")
        done
    done
done


for i in 2 4
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    for w_bits in 3
    do
        for seed in 0 1 2
        do
            max_length=8k
            q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
            job="eval cd /home/ylsung/codes/QuaRot/LEval; \
            python Baselines/general_instruct-test.py \
            --metric exam_eval \
            --max_length ${max_length} \
            --gpu 0 \
            --model_path mistralai/Mistral-Nemo-Instruct-2407 \
            --postfix _${q_type} \
            --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/mistral-nemo-instruct-2407_${q_type}.pth"

            echo $job
            jobs+=("$job")
        done
    done
done


for i in 2 4
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    for w_bits in 3
    do
        for seed in 0 1 2
        do
            max_length=8k
            q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
            folder=Predictions/exam_eval/mistral-nemo-instruct-2407-${max_length}_${q_type}
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


for round in 1 2
do
    for i in 2 4
    do
        nsamples=${nsamples_list[i]}
        train_seqlen=${train_seqlen_list[i]}
        for w_bits in 3
        do
            for seed in 0 1 2
            do
                q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
                job="eval cd /home/ylsung/codes/QuaRot/LongICLBench; \
                python my_tacred_infer_chat.py \
                --model mistral-nemo-instruct \
                --round ${round} \
                --test_number 500 \
                --postfix _chat_${q_type} \
                --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/mistral-nemo-instruct-2407_${q_type}.pth"

                echo $job
                jobs+=("$job")
            done
        done
    done
done


for round in 2 3
do
    for i in 2 4
    do
        nsamples=${nsamples_list[i]}
        train_seqlen=${train_seqlen_list[i]}
        for w_bits in 3
        do
            for seed in 0 1 2
            do
                q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
                job="eval cd /home/ylsung/codes/QuaRot/LongICLBench; \
                python my_banking77_infer_chat.py \
                --model mistral-nemo-instruct \
                --round ${round} \
                --test_number 500 \
                --postfix _chat_${q_type} \
                --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/mistral-nemo-instruct-2407_${q_type}.pth"

                echo $job
                jobs+=("$job")
            done
        done
    done
done


for i in 2 4
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    for w_bits in 3
    do
        for seed in 0 1 2
        do
            q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
            job="eval cd /home/ylsung/codes/QuaRot/lca-baselines/library_based_code_generation; \
            python -m src.evaluation.evaluate \
            --model mistralai/Mistral-Nemo-Instruct-2407 \
            --postfix _chat_${q_type} \
            --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/mistral-nemo-instruct-2407_${q_type}.pth"

            echo $job
            jobs+=("$job")
        done
    done
done

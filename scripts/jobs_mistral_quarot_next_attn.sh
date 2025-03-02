source /home/ylsung/codes/env.sh

nsamples_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
train_seqlen_list=(1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024 512 256)

# 3 to 12
# 3 6 9 12
nsamples_list=(16 64 256 1024 4096)
train_seqlen_list=(65536 16384 4096 1024 256)

declare -a jobs

model_name=mistralai/Mistral-Nemo-Instruct-2407
model_store_name=mistral-nemo-instruct-2407

for lr in 3e-5
do
    for config in attn1e-2_gd_scale
    do
        for i in 3
        do
            nsamples=${nsamples_list[i]}
            train_seqlen=${train_seqlen_list[i]}
            for w_bits in 4
            do
                for seed in 0 1 2
                do
                    q_type_postfix=next_${config}_lr${lr}
                    q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${q_type_postfix}@${seed}
                    save_name=${model_store_name}_${q_type}
                    # no rotate
                    # TODO: add --rotate back
                    job="eval cd /home/ylsung/codes/QuaRot/; \
                    python fake_quant/main.py \
                    --model ${model_name} \
                    --w_bits ${w_bits} --w_clip \
                    --compute_next_attn_loss \
                    --quant_lr ${lr} \
                    --rotate \
                    --optimizer_yaml fake_quant/configs/optimizer/${config}_lr1e-5.yaml \
                    --val_size 128 \
                    --seed ${seed} \
                    --nsamples ${nsamples} \
                    --train_seqlen ${train_seqlen} \
                    --wandb_project QuaRot \
                    --wandb_id ylsung \
                    --save_name ${save_name}  \
                    --lm_eval \
                    --save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
                    --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

                    echo $job
                    jobs+=("$job")
                done
            done
        done
    done
done

for lr in 3e-5
do
    for config in attn1e-2_gd_scale
    do
        for i in 3
        do
            nsamples=${nsamples_list[i]}
            train_seqlen=${train_seqlen_list[i]}
            for w_bits in 4
            do
                for seed in 0 1 2
                do
                    q_type_postfix=next_${config}_lr${lr}
                    q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${q_type_postfix}@${seed}
                    # long context testing
                    job="eval cd /home/ylsung/codes/QuaRot/qllm-eval/qllm_eval/evaluation/q_long/; \
                    python main_longeval_quarot.py \
                    --model-name-or-path ${model_name} --use_flash_attn \
                    --task lines --test_dir new_cases \
                    --sub_task 300 460 620 620_s0_e155 620_s156_e310 620_s311_e465 620_s466_e620 \
                    --prefix ${q_type}_ \
                    --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/${model_store_name}_${q_type}.pth"

                    echo $job
                    jobs+=("$job")
                done
            done
        done
    done
done

for lr in 3e-5
do
    for config in attn1e-2_gd_scale
    do
        for i in 3
        do
            nsamples=${nsamples_list[i]}
            train_seqlen=${train_seqlen_list[i]}
            for w_bits in 4
            do
                for seed in 0 1 2
                do
                    q_type_postfix=next_${config}_lr${lr}
                    max_length=8k
                    q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${q_type_postfix}@${seed}
                    job="eval cd /home/ylsung/codes/QuaRot/LEval; \
                    python Baselines/general_instruct-test.py \
                    --metric exam_eval \
                    --max_length ${max_length} \
                    --gpu 0 \
                    --model_path ${model_name} \
                    --postfix _${q_type} \
                    --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/${model_store_name}_${q_type}.pth"

                    echo $job
                    jobs+=("$job")
                done
            done
        done
    done
done


for lr in 3e-5
do
    for config in attn1e-2_gd_scale
    do
        for i in 3
        do
            nsamples=${nsamples_list[i]}
            train_seqlen=${train_seqlen_list[i]}
            for w_bits in 4
            do
                for seed in 0 1 2
                do
                    q_type_postfix=next_${config}_lr${lr}
                    max_length=8k
                    q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${q_type_postfix}@${seed}
                    folder=Predictions/exam_eval/${model_store_name}-${max_length}_${q_type}
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
done
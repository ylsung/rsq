source /home/ylsung/codes/env.sh

nsamples_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
train_seqlen_list=(1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024 512 256)

# 3 to 12
# 3 6 9 12
nsamples_list=(16 64 256 1024 4096)
train_seqlen_list=(65536 16384 4096 1024 256)

declare -a jobs

for i in 3
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    for w_bits in 3 4
    do
        for seed in 0 1 2
        do
            q_type=quarot_retrieval_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
            save_name=llama3-8b-instruct_${q_type}
            # no rotate
            
            job="eval cd /home/ylsung/codes/QuaRot/; \
            python fake_quant/main.py \
            --model meta-llama/Meta-Llama-3-8B-Instruct \
            --w_bits ${w_bits} --w_clip \
            --rotate \
            --seed ${seed} \
            --cal_dataset retrieval \
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

for i in 3
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    for w_bits in 3 4
    do
        for seed in 0 1 2
        do
            q_type=quarot_retrieval_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
            # long context testing
            job="eval cd /home/ylsung/codes/QuaRot/qllm-eval/qllm_eval/evaluation/q_long/; \
            python main_longeval_quarot.py \
            --model-name-or-path meta-llama/Meta-Llama-3-8B-Instruct --use_flash_attn \
            --task lines --test_dir new_cases \
            --sub_task 300 460 620 770 620_s0_e155 620_s156_e310 620_s311_e465 620_s466_e620 \
            --prefix ${q_type}_ \
            --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/llama3-8b-instruct_${q_type}.pth"

            echo $job
            jobs+=("$job")
        done
    done
done

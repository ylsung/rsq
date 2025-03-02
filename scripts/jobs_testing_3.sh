source /home/ylsung/codes/env.sh

nsamples_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
train_seqlen_list=(1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024 512 256)

# 3 to 12
# 3 6 9 12
nsamples_list=(128 256 512 1024 2048 8192 32768)
train_seqlen_list=(8192 4096 2048 1024 512 128 32)
val_size_list=(16 32 64 128 256 1024 4096)

declare -a jobs


for round in 1 2
do
    for i in 3
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
    for i in 3
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


for i in 3
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

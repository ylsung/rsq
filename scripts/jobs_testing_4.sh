source /home/ylsung/codes/env.sh

nsamples_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
train_seqlen_list=(1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024 512 256)

# 3 to 12
# 3 6 9 12
nsamples_list=(512 1024 2048 4096)
train_seqlen_list=(8192 4096 2048 1024)

declare -a jobs


q_types=(
    quarot_16bit_w
)

for round in 1 2
do
    for q_type in "${q_types[@]}"
    do
        job="eval cd /home/ylsung/codes/QuaRot/LongICLBench; \
        python my_tacred_infer_chat.py \
        --model llama3-8b-instruct \
        --round ${round} \
        --test_number 500 \
        --postfix _chat_${q_type} \
        --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/llama3-8b-instruct_${q_type}.pth"

        echo $job
        jobs+=("$job")
    done
done

q_type_prefixes=(
    quarot_3bit_w_n1024_l1024_next_attn_ratio100_gd_scale_lr1e-5_prob
    quarot_3bit_w_n1024_l1024_next_attn_ratio30_gd_scale_lr1e-5_prob
    quarot_3bit_w_n1024_l1024_next_attn_ratio10_gd_scale_lr1e-5_prob
    quarot_3bit_w_n1024_l1024_next_attn_ratio1_gd_scale_lr1e-5_prob
    quarot_3bit_w_n1024_l1024_next_attn_ratio0.1_gd_scale_lr1e-5_prob
    quarot_3bit_w_n1024_l1024_next_attn_ratio0.01_gd_scale_lr1e-5_prob
    quarot_3bit_w_n1024_l1024_next_attn_ratio1e-1_gd_scale_lr1e-5
    quarot_3bit_w_n1024_l1024_next_attn_ratio1e-2_gd_scale_lr1e-5
    quarot_3bit_w_n1024_l1024_next_attn_ratio30_gd_scale_lr1e-5_prob

    quarot_3bit_w_n4096_l1024
    quarot_3bit_w_n2048_l2048
    quarot_3bit_w_n1024_l4096
    quarot_3bit_w_n512_l8192

    quarot_3bit_w_n1024_l1024
    quarot_3bit_w_n512_l2048
    quarot_3bit_w_n256_l4096
    quarot_3bit_w_n128_l8192
)

for round in 1 2
do
    for q_type_prefix in "${q_type_prefixes[@]}"
    do
        for seed in 0 1 2
        do
            q_type=${q_type_prefix}@${seed}
            job="eval cd /home/ylsung/codes/QuaRot/LongICLBench; \
            python my_tacred_infer_chat.py \
            --model llama3-8b-instruct \
            --round ${round} \
            --test_number 500 \
            --postfix _chat_${q_type} \
            --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/llama3-8b-instruct_${q_type}.pth"

            echo $job
            jobs+=("$job")
        done
    done
done

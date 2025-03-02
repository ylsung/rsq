# gpu_index=$1
# for q_type in w_n1024_l1024 w_n4096_l256 w_n256_l4096 w_n64_l16384 # w_n16_l65536
# do
#     echo run on ${gpu_index}th GPU
#     CUDA_VISIBLE_DEVICES=${gpu_index} python Baselines/llama3-instruct-test.py \
#     --metric exam_eval \
#     --max_length 8k \
#     --gpu 0 \
#     --postfix _${q_type} \
#     --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/llama3-8b-instruct_quarot_4bit_${q_type}.pth &
#     gpu_index=$(($gpu_index+1))
# done
# --task_name quality \

# --postfix _${q_type} \
# --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/quarot_4bit_${q_type}.pth


gpu_index=$1

for max_length in 8k 12k 16k 20k
do
    # 16 bits
    echo run on ${gpu_index}th GPU
    CUDA_VISIBLE_DEVICES=${gpu_index} python Baselines/llama3-instruct-test.py \
    --metric exam_eval \
    --max_length ${max_length} \
    --gpu 0 &
    gpu_index=$(($gpu_index+1))
done

    # for q_type in w_n16_l65536
    # do
    #     echo run on ${gpu_index}th GPU
    #     CUDA_VISIBLE_DEVICES=${gpu_index} python Baselines/llama3-instruct-test.py \
    #     --metric exam_eval \
    #     --max_length 8k \
    #     --gpu 0 \
    #     --postfix _${q_type} \
    #     --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/llama3-8b-instruct_quarot_4bit_${q_type}.pth &
    #     gpu_index=$(($gpu_index+1))
    # done
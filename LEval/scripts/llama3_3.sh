gpu_index=$1

q_type="2bit_w_n1024_l1024"
for max_length in 8k 12k 16k 20k
do
    # 16 bits
    echo run on ${gpu_index}th GPU
    CUDA_VISIBLE_DEVICES=${gpu_index} python Baselines/llama3-instruct-test.py \
    --metric exam_eval \
    --max_length ${max_length} \
    --gpu 0 \
    --postfix _${q_type} \
    --quantized_checkpoint /nas-hdd/ylsung/checkpoints/llama3-8b-instruct_${q_type}.pth &
    gpu_index=$(($gpu_index+1))
done

q_type="3bit_w_n1024_l1024"
for max_length in 8k 12k 16k 20k
do
    # 16 bits
    echo run on ${gpu_index}th GPU
    CUDA_VISIBLE_DEVICES=${gpu_index} python Baselines/llama3-instruct-test.py \
    --metric exam_eval \
    --max_length ${max_length} \
    --gpu 0 \
    --postfix _${q_type} \
    --quantized_checkpoint /nas-hdd/ylsung/checkpoints/llama3-8b-instruct_${q_type}.pth &
    gpu_index=$(($gpu_index+1))

done
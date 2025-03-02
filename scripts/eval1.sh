source /home/ylsung/codes/env.sh

# nsamples_list=(32 64 128 256 512 1024 2048 4096)
# train_seqlen_list=(32768 16384 8192 4096 2048 1024 512 256)

nsamples_list=(1024)
train_seqlen_list=(1024)

gpu_index=$1
for i in 3 4
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    save_name="llama3-8b_4bit_w_n"${nsamples}_l${train_seqlen}
    # no rotate

    echo using GPU ${gpu_index} on ${save_name}

    WANDB_API_KEY=${WANDB_API_KEY} CUDA_VISIBLE_DEVICES=${gpu_index} python fake_quant/main_llama3_intruct.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --w_bits 4 --w_clip \
    --wandb \
    --nsamples ${nsamples} \
    --train_seqlen ${train_seqlen} \
    --wandb_project QuaRot \
    --wandb_id ylsung \
    --save_name ${save_name} \
    --lm_eval \
    --tasks "piqa" "hellaswag" "arc_easy" "arc_challenge" "winogrande" "lambada" &

    gpu_index=$(($gpu_index+1))
done

for i in 3 4
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    save_name="llama3-8b_4bit_rotate_w_n"${nsamples}_l${train_seqlen}

    echo using GPU ${gpu_index} on ${save_name}

    WANDB_API_KEY=${WANDB_API_KEY} CUDA_VISIBLE_DEVICES=${gpu_index} python fake_quant/main.py \
    --model meta-llama/Meta-Llama-3-8B \
    --w_bits 4 --w_clip \
    --rotate \
    --wandb \
    --nsamples ${nsamples} \
    --train_seqlen ${train_seqlen} \
    --wandb_project QuaRot \
    --wandb_id ylsung \
    --save_name ${save_name} \
    --lm_eval \
    --tasks "piqa" "hellaswag" "arc_easy" "arc_challenge" "winogrande" "lambada" &

    gpu_index=$(($gpu_index+1))
done

# WANDB_API_KEY=${WANDB_API_KEY} CUDA_VISIBLE_DEVICES=${1} python fake_quant/main.py \
    # --model meta-llama/Meta-Llama-3-8B \
    # --lm_eval \
    # --wandb \
    # --wandb_project QuaRot \
    # --wandb_id ylsung \
    # --save_name llama3-8b_16bit \
    # --tasks "piqa" "hellaswag" "arc_easy" "arc_challenge" "winogrande" "lambada"

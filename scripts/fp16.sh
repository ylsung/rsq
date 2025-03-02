source /home/ylsung/codes/env.sh

# nsamples_list=(32 64 128 256 512 1024 2048 4096)
# train_seqlen_list=(32768 16384 8192 4096 2048 1024 512 256)

gpu_index=$1

save_name="llama3-8b-instruct_16bit"
# no rotate

echo using GPU ${gpu_index} on ${save_name}

WANDB_API_KEY=${WANDB_API_KEY} CUDA_VISIBLE_DEVICES=${gpu_index} python fake_quant/main.py \
--model meta-llama/Meta-Llama-3-8B-Instruct \
--wandb \
--wandb_project QuaRot \
--wandb_id ylsung \
--save_name ${save_name} \
--lm_eval \
--tasks "piqa" "hellaswag" "arc_easy" "arc_challenge" "winogrande" "lambada"

source /home/ylsung/codes/env.sh

# nsamples_list=(32 64 128 256 512 1024 2048 4096)
# train_seqlen_list=(32768 16384 8192 4096 2048 1024 512 256)

nsamples_list=(1024)
train_seqlen_list=(1024)

gpu_index=$1

w_bits=16
save_name=llama3-8b-instruct_quarot_${w_bits}bit_w

echo using GPU ${gpu_index} on ${save_name}

WANDB_API_KEY=${WANDB_API_KEY} CUDA_VISIBLE_DEVICES=${gpu_index} python fake_quant/main.py \
--model meta-llama/Meta-Llama-3-8B-Instruct \
--w_bits ${w_bits} --w_clip \
--rotate \
--wandb_project QuaRot \
--wandb_id ylsung \
--save_name ${save_name} \
--lm_eval \
--save_qmodel_path /nas-hdd/ylsung/checkpoints/llama3-8b-instruct_quarot_${w_bits}bit_w.pth \
--tasks "piqa" "hellaswag" "arc_easy" "arc_challenge" "winogrande" "lambada"

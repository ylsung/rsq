source /home/ylsung/codes/env.sh

nsamples_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
train_seqlen_list=(1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024 512 256)

# 3 to 12
# 3 6 9 12
nsamples_list=(16 64 256 1024 4096)
train_seqlen_list=(65536 16384 4096 1024 256)

declare -a jobs

for w_bits in 3 4
do
    q_type=quarot_${w_bits}bit_w_rtn
    save_name=llama3-8b-instruct_${q_type}
    # no rotate
    
    job="eval cd /home/ylsung/codes/QuaRot/; \
    python fake_quant/main.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --w_bits ${w_bits} --w_clip \
    --rotate \
    --w_rtn \
    --wandb_project QuaRot \
    --wandb_id ylsung \
    --save_name ${save_name} \
    --lm_eval \
    --save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
    --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

    echo $job
    jobs+=("$job")
done
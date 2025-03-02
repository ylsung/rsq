source /home/ylsung/codes/env.sh

nsamples_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
train_seqlen_list=(1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024 512 256)

# 3 to 12
# 3 6 9 12
nsamples_list=(128 256 512 1024 2048 8192 4096 16384 32768)
train_seqlen_list=(8192 4096 2048 1024 512 128 256 64 32)
val_size_list=(16 32 64 128 256 1024 512 2048 4096)

declare -a jobs

seeds=(0 1 2)

source scripts/additional_eval_func.sh

i=2

for i in 2
do
    for masking in 0.5
    do
        nsamples=${nsamples_list[i]}
        train_seqlen=${train_seqlen_list[i]}
        val_size=${val_size_list[i]}
        for w_bits in 3
        do
            for attn_weighting in input_magnitude
            do
                for seed in 0
                do
                    q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}_masking${masking}@${seed}
                    save_name=llama3-8b-instruct_${q_type}
                    # no rotate

                    job="eval cd /home/ylsung/codes/QuaRot/; \
                    python fake_quant/main.py \
                    --model meta-llama/Meta-Llama-3-8B-Instruct \
                    --w_bits ${w_bits} --w_clip \
                    --rotate \
                    --seed ${seed} \
                    --val_size ${val_size} \
                    --masking ${masking} \
                    --add_until_fail \
                    --module_input_weighting_yaml fake_quant/configs/input_weighting/${attn_weighting}.yaml \
                    --nsamples ${nsamples} \
                    --train_seqlen ${train_seqlen} \
                    --wandb_project QuaRot \
                    --wandb_id ylsung \
                    --save_name ${save_name} \
                    --lm_eval \
                    --load_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
                    --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada"

                    echo $job
                    jobs+=("$job")
                done
            done
            
        done
    done
done
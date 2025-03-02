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


nsamples=${nsamples_list[i]}
train_seqlen=${train_seqlen_list[i]}
val_size=${val_size_list[i]}

# 0_4: 512
# 0_8: 256
# 0_16: 128
# 0_32: 64
# 0_64: 32

# 0_3_4: 1024
# 0_7_8: 512
# 0_15_16: 256
# 0_31_32: 128
# 0_63_64: 64
# 0_127_128: 32

for w_bits in 3
do
    for attn_weighting in "0_31_32"
    do
        for seed in ${seeds[@]}
        do
            q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_masking${attn_weighting}_expand8@${seed}
            save_name=llama3-8b-instruct_${q_type}
            # no rotate
            
            job="eval cd /home/ylsung/codes/QuaRot/; \
            python fake_quant/main.py \
            --model meta-llama/Meta-Llama-3-8B-Instruct \
            --w_bits ${w_bits} --w_clip \
            --seed ${seed} \
            --val_size ${val_size} \
            --rotate \
            --module_input_weighting_yaml fake_quant/configs/input_weighting/masking_first.yaml \
            --adhoc_weighting_method_type ${attn_weighting} \
            --train_seqlen ${train_seqlen} \
            --nsamples ${nsamples} \
            --expand_factor 8 \
            --offload_activations \
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


for w_bits in 3
do
    for attn_weighting in "0_31_32"
    do
        for seed in ${seeds[@]}
        do
            q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_masking${attn_weighting}_expand8@${seed}
            save_name=llama3-8b-instruct_${q_type}
            # no rotate
            
            mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name})

            for item in "${new_jobs[@]}"; do
                echo $item
            done

            jobs+=("${new_jobs[@]}")
            
        done
    done
done

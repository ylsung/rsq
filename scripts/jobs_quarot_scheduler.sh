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

i=3


nsamples=${nsamples_list[i]}
train_seqlen=${train_seqlen_list[i]}
val_size=${val_size_list[i]}
max_value=5
w_bits=3

for factor in 1e-1 3e-2
do
    for attn_weighting in "startpeak"
    do
        for seed in ${seeds[@]}
        do
            q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}${max_value}_factor${factor}@${seed}
            save_name=llama3-8b-instruct_${q_type}
            # no rotate
            
            job="eval cd /home/ylsung/codes/QuaRot/; \
            python fake_quant/main.py \
            --model meta-llama/Meta-Llama-3-8B-Instruct \
            --w_bits ${w_bits} --w_clip \
            --seed ${seed} \
            --val_size ${val_size} \
            --rotate \
            --max_value ${max_value} \
            --factor ${factor} \
            --scheduler_yaml fake_quant/configs/schedulers/${attn_weighting}.yaml \
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


for factor in 30
do
    for attn_weighting in "endpointspeak"
    do
        for seed in ${seeds[@]}
        do
            q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}${max_value}_factor${factor}@${seed}
            save_name=llama3-8b-instruct_${q_type}
            # no rotate
            
            job="eval cd /home/ylsung/codes/QuaRot/; \
            python fake_quant/main.py \
            --model meta-llama/Meta-Llama-3-8B-Instruct \
            --w_bits ${w_bits} --w_clip \
            --seed ${seed} \
            --val_size ${val_size} \
            --rotate \
            --max_value ${max_value} \
            --factor ${factor} \
            --scheduler_yaml fake_quant/configs/schedulers/${attn_weighting}.yaml \
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


for factor in 1e-1 3e-2
do
    for attn_weighting in "startpeak"
    do
        for seed in ${seeds[@]}
        do
            q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}${max_value}_factor${factor}@${seed}
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


for factor in 30
do
    for attn_weighting in "endpointspeak"
    do
        for seed in ${seeds[@]}
        do
            q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${attn_weighting}${max_value}_factor${factor}@${seed}
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

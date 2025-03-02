source /home/ylsung/codes/env.sh

nsamples_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
train_seqlen_list=(1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024 512 256)

# 3 to 12
# 3 6 9 12
nsamples_list=(16 64 256 1024 4096)
train_seqlen_list=(65536 16384 4096 1024 256)

gpu_index=$1

declare -a jobs

for i in 3
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}

    w_bits=4
    for layer in 0 10 20 30
    do
        save_name=llama3-8b-instruct_quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_${layer}th_layer_noquantize
        # no rotate

        job="python fake_quant/main.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --w_bits ${w_bits} --w_clip \
        --rotate \
        --nsamples ${nsamples} \
        --train_seqlen ${train_seqlen} \
        --wandb_project QuaRot \
        --wandb_id ylsung \
        --save_name ${save_name} \
        --lm_eval \
        --save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
        --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada \
        --layers_dont_quantize ${layer}"

        echo $job

        jobs+=("$job")

        # save_name=llama3-8b-instruct_quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_first_attn_noquantize
        # # no rotate

        # job="python fake_quant/main.py \
        # --model meta-llama/Meta-Llama-3-8B-Instruct \
        # --w_bits ${w_bits} --w_clip \
        # --rotate \
        # --nsamples ${nsamples} \
        # --train_seqlen ${train_seqlen} \
        # --wandb_project QuaRot \
        # --wandb_id ylsung \
        # --save_name ${save_name} \
        # --lm_eval \
        # --save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
        # --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada \
        # --layers_dont_quantize 0 \
        # --dont_quantize_attn"

        # echo $job

        # jobs+=("$job")

        # save_name=llama3-8b-instruct_quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_first_qk_noquantize
        # # no rotate

        # job="python fake_quant/main.py \
        # --model meta-llama/Meta-Llama-3-8B-Instruct \
        # --w_bits ${w_bits} --w_clip \
        # --rotate \
        # --nsamples ${nsamples} \
        # --train_seqlen ${train_seqlen} \
        # --wandb_project QuaRot \
        # --wandb_id ylsung \
        # --save_name ${save_name} \
        # --lm_eval \
        # --save_qmodel_path /nas-hdd/ylsung/checkpoints/${save_name}.pth \
        # --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada \
        # --layers_dont_quantize 0 \
        # --dont_quantize_qk"

        # echo $job

        # jobs+=("$job")
    done
done

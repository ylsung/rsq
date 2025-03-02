source /home/ylsung/codes/env.sh

nsamples_list=(16 64 256 1024 4096)
train_seqlen_list=(65536 16384 4096 1024 256)

declare -a jobs


for i in 3
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    for w_bits in 3 4
    do
        for seed in 0
        do
            for max_length in 8k
            do
                q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_org_attn_none3
                job="eval cd /home/ylsung/codes/QuaRot/LEval; \
                python Baselines/llama3-instruct-test.py \
                --metric exam_eval \
                --max_length ${max_length} \
                --gpu 0 \
                --postfix _${q_type} \
                --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/llama3-8b-instruct_${q_type}.pth"

                echo $job
                jobs+=("$job")
            done
        done
    done
done



for i in 3
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    for w_bits in 3 4
    do
        for seed in 1 2 3 4
        do
            for max_length in 8k
            do
                q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_org_attn_none3@${seed}
                job="eval cd /home/ylsung/codes/QuaRot/LEval; \
                python Baselines/llama3-instruct-test.py \
                --metric exam_eval \
                --max_length ${max_length} \
                --gpu 0 \
                --postfix _${q_type} \
                --rotated_quantized_checkpoint /nas-hdd/ylsung/checkpoints/llama3-8b-instruct_${q_type}.pth"

                echo $job
                jobs+=("$job")
            done
        done
    done
done


for i in 3
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    for w_bits in 3 4
    do
        for seed in 0
        do
            for max_length in 8k
            do
                q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_org_attn_none3
                folder=Predictions/exam_eval/llama3-8B-instruct-${max_length}_${q_type}
                for task in tpo quality coursera sci_fi gsm100 codeU topic_retrieval_longchat
                do
                    job="eval cd /home/ylsung/codes/QuaRot/LEval; \
                    python Evaluation/auto_eval.py \
                    --pred_file ${folder}/${task}.pred.jsonl"

                    echo $job
                    jobs+=("$job")
                done
            done

        done
    done
done


for i in 3
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    for w_bits in 3 4
    do
        for seed in 1 2 3 4
        do
            for max_length in 8k
            do
                q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}_org_attn_none3@${seed}
                folder=Predictions/exam_eval/llama3-8B-instruct-${max_length}_${q_type}
                for task in tpo quality coursera sci_fi gsm100 codeU topic_retrieval_longchat
                do
                    job="eval cd /home/ylsung/codes/QuaRot/LEval; \
                    python Evaluation/auto_eval.py \
                    --pred_file ${folder}/${task}.pred.jsonl"

                    echo $job
                    jobs+=("$job")
                done
            done

        done
    done
done


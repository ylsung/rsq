
for q_type in quarot_2bit quarot_3bit
do
    for max_length in 8k 12k 16k 20k
    do
    folder=Predictions/exam_eval/llama3-8B-instruct-${max_length}_${q_type}_w_n1024_l1024
    bash scripts/eval.sh ${folder}
    done
done

# for max_length in 8k 12k 16k 20k
# do
# folder=Predictions/exam_eval/llama3-8B-instruct-${max_length}
# bash scripts/eval.sh ${folder}
# done
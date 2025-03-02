
folder=$1


for task in tpo quality coursera sci_fi gsm100 codeU topic_retrieval_longchat
do
    echo ${task}
    python Evaluation/auto_eval.py \
        --pred_file ${folder}/${task}.pred.jsonl
done

python Evaluation/read_result.py ${folder}
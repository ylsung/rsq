# for testing purposes
source $2


for job in "${jobs[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 ${job}
done
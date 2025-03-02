source /home/ylsung/codes/env.sh

nsamples_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
train_seqlen_list=(1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024 512 256)

# 3 to 12
# 3 6 9 12
nsamples_list=(16 64 256 1024 4096)
train_seqlen_list=(65536 16384 4096 1024 256)

declare -a jobs
val_seqlen=16384

for i in 0 1 2 3 4
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    for w_bits in 3 4
    do
        save_name="llama3-8b-instruct_quarot_${w_bits}bit_w_n"${nsamples}_l${train_seqlen}
        # no rotate

        job="python fake_quant/wiki_eval.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --w_bits ${w_bits} --w_clip \
        --rotate \
        --nsamples ${nsamples} \
        --train_seqlen ${train_seqlen} \
        --wandb_project QuaRot \
        --wandb_id ylsung \
        --save_name ${save_name} \
        --bsz 1 \
        --val_seqlen ${val_seqlen} \
        --load_qmodel_path /nas-hdd/ylsung/checkpoints/llama3-8b-instruct_quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}.pth"

        echo $job
        jobs+=("$job")
    done

    for w_bits in 3 4
    do
        save_name="llama3-8b-instruct_${w_bits}bit_w_n"${nsamples}_l${train_seqlen}
        # no rotate

        job="python fake_quant/wiki_eval.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --w_bits ${w_bits} --w_clip \
        --nsamples ${nsamples} \
        --train_seqlen ${train_seqlen} \
        --wandb_project QuaRot \
        --wandb_id ylsung \
        --save_name ${save_name} \
        --bsz 1 \
        --val_seqlen ${val_seqlen} \
        --load_qmodel_path /nas-hdd/ylsung/checkpoints/llama3-8b-instruct_${w_bits}bit_w_n${nsamples}_l${train_seqlen}.pth"

        echo $job
        jobs+=("$job")
    done
done


# Total number of GPUs available
TOTAL_GPUS=8

# Dynamically create an array to keep track of GPU usage status
declare -a gpu_locks
for i in $(seq 0 $((TOTAL_GPUS - 1))); do
    gpu_locks+=("0")  # Initially, all GPUs are unlocked (0)
done

# Array containing your jobs

# for i in 1 2 3 4 5 6 7 8 9 10; do
#     job="python \
#     scripts/test.py \
#     ${i}${i} fasfds \
#     ddd"
#     # echo "${job}" is added
#     jobs+=("$job")
# done

MEMORY_THRESHOLD=50
# Function to get a list of free GPU indices
function get_free_gpus {
  local free_gpus=()
  
  # Get the current memory usage of GPUs
  local gpu_memory_usage=($(gpustat | sed 's/\x1B\[[0-9;]\+[A-Za-z]//g' | grep -oP '\d+ / \d+ MB' | awk '{print $1}'))

  # Check each GPU to determine if it's free
  for gpu_index in $(seq 0 $((TOTAL_GPUS - 1))); do
    if [ ${gpu_memory_usage[$gpu_index]} -lt $MEMORY_THRESHOLD ] && [ ${gpu_locks[$gpu_index]} -eq 0 ]; then
      free_gpus+=($gpu_index)
    fi
  done

  echo "${free_gpus[@]}"
}

declare -a job_pids

# Run jobs on available GPUs
job_index=0
while [ $job_index -lt ${#jobs[@]} ]; do
  free_gpus=($(get_free_gpus))
  while [ ${#free_gpus[@]} -eq 0 ]; do
    # Check running jobs and update locks
    for pid in "${!job_pids[@]}"; do
      if ! kill -0 $pid 2>/dev/null; then
        # Job has finished, release GPU
        gpu_locks[${job_pids[$pid]}]=0
        unset job_pids[$pid]
      fi
    done
    sleep 20  # Re-check every 20 seconds
    free_gpus=($(get_free_gpus))
  done
  
  # Use the first free GPU
  selected_gpu=${free_gpus[0]}
  gpu_locks[$selected_gpu]=1  # Lock this GPU
  echo "Starting job $job_index on GPU $selected_gpu: ${jobs[$job_index]}"
  WANDB_API_KEY=${WANDB_API_KEY} CUDA_VISIBLE_DEVICES=$selected_gpu ${jobs[$job_index]} &
  job_pid=$!
  job_pids[$job_pid]=$selected_gpu
  
  # Increment job index
  ((job_index++))
done

# Wait for all jobs to complete
for pid in "${!job_pids[@]}"; do
  wait $pid
  gpu_locks[${job_pids[$pid]}]=0
done
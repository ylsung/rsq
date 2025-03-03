source /home/ylsung/codes/env.sh

# Total number of GPUs available
TOTAL_GPUS=4
AVAILABLE_GPUS=None

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --total_gpus) TOTAL_GPUS="$2"; shift ;;
    --available) AVAILABLE_GPUS="$2"; shift ;;
    --jobs) JOBS="$2"; shift ;;
    -t) TOTAL_GPUS="$2"; shift ;;
    -a) AVAILABLE_GPUS="$2"; shift ;;
    -j) JOBS="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# Use all GPUS if AVAILABLE_GPUS is not assigned
if [ "$AVAILABLE_GPUS" == "None" ]; then
  AVAILABLE_GPUS=$(seq -s, 0 $(($TOTAL_GPUS-1)))
fi
max_gpu_id=$(echo "$AVAILABLE_GPUS" | tr ',' '\n' | sort -n | tail -1)

if [ "$max_gpu_id" -ge ${TOTAL_GPUS} ]; then
    echo "The maximum GPU id (=${max_gpu_id}) should less ${TOTAL_GPUS}";
    exit 1;
fi

echo TOTAL_GPUS=$TOTAL_GPUS
echo AVAILABLE_GPUS=$AVAILABLE_GPUS

# Declare the jobs 
source ${JOBS}

# Dynamically create an array to keep track of GPU usage status
declare -a gpu_locks
declare -a available_gpus

for i in $(seq 0 $((TOTAL_GPUS - 1))); do
    gpu_locks+=("0")  # Initially, all GPUs are unlocked (0)
    available_gpus+=("0")  # Initialized all GPUs to be unavailable 
done

# Convert the input string to an array of indices
IFS=',' read -r -a available_gpus_indices <<< "$AVAILABLE_GPUS"

# Set the corresponding indices to 1
for index in "${available_gpus_indices[@]}"; do
  available_gpus[$((index))]=1
done

echo "AVAILABLE_GPUS_LIST=${available_gpus[@]}"

# Array containing your jobs

# for i in 1 2 3 4 5 6 7 8 9 10; do
#     job="python \
#     scripts/test.py \
#     ${i}${i} fasfds \
#     ddd"
#     # echo "${job}" is added
#     jobs+=("$job")
# done

MEMORY_THRESHOLD=150
# Function to get a list of free GPU indices
function get_free_gpus {
  local free_gpus=()
  
  # Get the current memory usage of GPUs
  local gpu_memory_usage=($(gpustat | sed 's/\x1B\[[0-9;]\+[A-Za-z]//g' | grep -oP '\d+ / \d+ MB' | awk '{print $1}'))

  # Check each GPU to determine if it's free
  for gpu_index in $(seq 0 $((TOTAL_GPUS - 1))); do
    if [ ${gpu_memory_usage[$gpu_index]} -lt $MEMORY_THRESHOLD ] && [ ${available_gpus[$gpu_index]} -eq 1 ] && [ ${gpu_locks[$gpu_index]} -eq 0 ]; then
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
    sleep 100  # Re-check every 100 seconds
    free_gpus=($(get_free_gpus))
  done

  echo "${free_gpus[@]}"
  
  # Use the first free GPU
  selected_gpu=${free_gpus[0]}
  gpu_locks[$selected_gpu]=1  # Lock this GPU
  echo "Starting job $job_index on GPU $selected_gpu: ${jobs[$job_index]}"
  CUDA_VISIBLE_DEVICES=$selected_gpu ${jobs[$job_index]} &
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
#!/bin/bash

# Total number of GPUs available
TOTAL_GPUS=4

# Array containing your jobs

declare -a jobs

for i in 1 2 3 4 5 6 7 8 9 10; do
    job="python \
    scripts/test.py \
    ${i} fasfds \
    ddd"
    # echo "${job}" is added
    jobs+=("$job")
done

MEMORY_THRESHOLD=100
# Function to get an array of free GPU IDs
function get_free_gpus {
  # Get GPU memory usage from gpustat and strip ANSI color codes
  local gpu_memory_usage=($(gpustat | sed 's/\x1B\[[0-9;]\+[A-Za-z]//g' | grep -oP '\d+ / \d+ MB' | awk '{print $1}'))

  # Initialize an array to hold the indices of free GPUs
  local free_gpus=()
  
  # Check each GPU's memory usage to determine if it's free
  for gpu_index in ${!gpu_memory_usage[@]}; do
    if [ ${gpu_memory_usage[$gpu_index]} -lt $MEMORY_THRESHOLD ]; then
      free_gpus+=($gpu_index)
    fi
  done

  echo "${free_gpus[@]}"
}
# Run jobs on available GPUs
job_index=0
while [ $job_index -lt ${#jobs[@]} ]; do
  free_gpus=($(get_free_gpus))
  while [ ${#free_gpus[@]} -eq 0 ]; do
    sleep 10  # Re-check every 10 seconds
    free_gpus=($(get_free_gpus))
  done

  echo "Free GPUs: "$(get_free_gpus)
  
  # Use the first free GPU
  echo "Starting job $job_index on GPU ${free_gpus[0]}: ${jobs[$job_index]}"
  CUDA_VISIBLE_DEVICES=${free_gpus[0]} ${jobs[$job_index]} &
  
  # Increment job index
  ((job_index++))
done

# Wait for all jobs to complete
wait
echo "All jobs have completed."

# # Function to get the number of GPUs currently in use
# function get_used_gpus {
#   # Check how many GPUs are currently being used by checking process list in nvidia-smi
#   nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | wc -l
# }

# # Run jobs on available GPUs
# job_index=0
# while [ $job_index -lt ${#jobs[@]} ]; do
#   # Check for available GPU
#   while [ $(get_used_gpus) -ge $TOTAL_GPUS ]; do
#     # Wait for a GPU to become available
#     sleep 10  # Checks every 10 seconds
#   done
  
#   # Run job and print the job and GPU index
#   echo "Starting job $job_index: ${jobs[$job_index]} running on $(($job_index % $TOTAL_GPUS))"
#   CUDA_VISIBLE_DEVICES=$(($job_index % $TOTAL_GPUS)) ${jobs[$job_index]} &
  
#   # Increment job index
#   ((job_index++))
# done

# # Wait for all jobs to complete
# wait
# echo "All jobs have completed."

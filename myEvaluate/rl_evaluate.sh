#!/bin/bash

# Available GPUs
cuda_devices=(2 3)

models=(
    "CodeDPO/qwen25-coder-inst-7b-reinforce-plus_v2_mini_processed_r1"
)


splits=(
  "complete"
  "instruct"
)

subsets=(
  "hard"
  "full"
)

# We’ll store all commands in an array
declare -a commands=()


for model in ${models[@]}
do
  for split in ${splits[@]}
  do
    for subset in ${subsets[@]}
    do
      commands+=( \
        "python myEvaluate/rl_evaluate.py \
          --model $model \
          --split $split \
          --subset $subset" \
      )
    done
  done
done


# -------------------------------------------------------------------
#               3. Concurrency & GPU handling
# -------------------------------------------------------------------
#
# We will keep track of which GPU each PID is using, so we can free
# that GPU after the PID finishes.

# Concurrency limit (e.g., up to # of GPUs, or set a smaller/larger number)
max_concurrent=${#cuda_devices[@]}  # default to number of GPUs

# The array of *currently available* GPUs. We pop from here when we start a job,
# and push back when the job finishes.
available_gpus=("${cuda_devices[@]}")

# Associative array that maps PID -> GPU device
declare -A pid_to_gpu

#
# 3.1) A helper function that attempts to free any finished jobs:
#      - Uses `wait -n` to wait for the *next* job to finish.
#      - Once a job finishes, we do `available_gpus+=(that_gpu)`.
#      - Repeat as many times as needed to keep concurrency under control.
#
clean_up_finished_jobs() {
    local current_jobs
    current_jobs=$(jobs -p | wc -l)

    # As long as we are at or above the concurrency limit, or as long as
    # we have 0 available GPUs, we want to wait for a job to finish.
    while [ "$current_jobs" -ge "$max_concurrent" ] || [ ${#available_gpus[@]} -eq 0 ]; do
        
        # Wait for any single job to finish
        wait -n 2>/dev/null  # The exit code of that job is stored in $?
        
        # Now, find which PIDs have exited, free up their GPU
        # We'll iterate over known PIDs in pid_to_gpu to see which ended.
        for p in "${!pid_to_gpu[@]}"; do
            # If the process is no longer running:
            if ! kill -0 "$p" 2>/dev/null; then
                # Freed GPU
                available_gpus+=("${pid_to_gpu[$p]}")
                unset pid_to_gpu["$p"]
            fi
        done
        
        current_jobs=$(jobs -p | wc -l)
        
        # If concurrency is now below limit and we have free GPUs, break
        if [ "$current_jobs" -lt "$max_concurrent" ] && [ ${#available_gpus[@]} -gt 0 ]; then
            break
        fi
        
        # Otherwise, keep trying to wait for more to finish
        # (In practice, one wait -n per loop is often enough)
    done
}

#
# 3.2) Launch one command with a free GPU
#
launch_command() {
    local cmd="$1"
    
    # Pop the first GPU from the available list
    local gpu="${available_gpus[0]}"
    available_gpus=("${available_gpus[@]:1}")

    # Start the command in the background
    # (We quote everything carefully to avoid issues with multiple args)
    CUDA_VISIBLE_DEVICES="$gpu" bash -c "$cmd" &

    # Record the PID->GPU in our map
    local pid=$!
    pid_to_gpu[$pid]="$gpu"
}

#
# 3.3) Main loop: For each command, wait if necessary, then launch
#
for cmd in "${commands[@]}"; do
    # If we have no free GPUs or we're already at concurrency limit, wait for a job to finish
    while [ ${#available_gpus[@]} -eq 0 ] || [ "$(jobs -p | wc -l)" -ge "$max_concurrent" ]; do
        clean_up_finished_jobs
    done
    
    # Now we have a free GPU and are below concurrency limit => launch command
    launch_command "$cmd"
done

#
# 3.4) After launching everything, wait for all remaining jobs to finish
#
while [ "$(jobs -p | wc -l)" -gt 0 ]; do
    clean_up_finished_jobs
done

echo "All tasks completed."
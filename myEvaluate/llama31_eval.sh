#!/bin/bash
# export BIGCODEBENCH_TIMEOUT_PER_TASK=10

model="meta-llama/Llama-3.1-8B-Instruct"

splits=(
  "complete"
  "instruct"
)

subsets=(
  "hard"
  "full"
)

n_lst=(
  # 16
  # 32
  64
)

for n in "${n_lst[@]}"
do
    # Inside this process, loop over splits and subsets
    for split in "${splits[@]}"
    do
      for subset in "${subsets[@]}"
      do
        # You can adjust or uncomment your commands here as needed.
        # For example, if you only want to do eval, do:
        python myEvaluate/myEvaluate.py \
          --model "${model}" \
          --split "${split}" \
          --subset "${subset}" \
          --n_samples "${n}" \
          --do_eval True &
      done
    done
done


wait
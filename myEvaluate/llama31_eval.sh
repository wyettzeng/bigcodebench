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

python myEvaluate/myEvaluate.py \
  --model "${model}" \
  --split "instruct" \
  --subset "full" \
  --n_samples 64 \
  --do_eval True

wait
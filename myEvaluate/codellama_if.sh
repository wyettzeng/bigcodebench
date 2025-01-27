#!/bin/bash
export BIGCODEBENCH_TIMEOUT_PER_TASK=10
model="codellama/CodeLlama-7b-Instruct-hf"

# python myEvaluate/myEvaluate.py \
#   --model ${model} \
#   --split "instruct" \
#   --subset "full" \
#   --n_samples 64 \
#   --do_eval False \
#   --tp 4 \
#   --bs 8

python myEvaluate/myEvaluate.py \
  --model ${model} \
  --split "instruct" \
  --subset "full" \
  --n_samples 64 \
  --do_eval True > codellama_if_64.log
#!/bin/bash
# export BIGCODEBENCH_TIMEOUT_PER_TASK=10
model="meta-llama/Llama-3.1-8B-Instruct"

# we only need to evaluate instruct hard version with 64 inferences as that's the one errored out

# python myEvaluate/myEvaluate.py \
#   --model ${model} \
#   --split "complete" \
#   --subset "full" \
#   --n_samples 64 \
#   --do_eval False \
#   --tp 4 \
#   --bs 8

python myEvaluate/myEvaluate.py \
  --model ${model} \
  --split "complete" \
  --subset "full" \
  --n_samples 64 \
  --do_eval True
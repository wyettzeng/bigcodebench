#!/bin/bash
model="meta-llama/Llama-3.1-8B-Instruct"

python myEvaluate/myEvaluate.py \
  --model ${model} \
  --split "instruct" \
  --subset "full" \
  --n_samples 64 \
  --do_eval False \
  --tp 2 \
  --bs 8

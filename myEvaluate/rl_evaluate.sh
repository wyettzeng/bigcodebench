#!/bin/bash

models=(
  "Qwen2.5-Coder-7B-Instruct"
  "checkpoint-200"
  "checkpoint-400"
  "checkpoint-600"
  "checkpoint-800"
)

splits=(
  "complete"
  "instruct"
)

subsets=(
  "hard"
  "full"
)


for model in ${models[@]}
do
  for split in ${splits[@]}
  do
    for subset in ${subsets[@]}
    do
        python myEvaluate/rl_evaluate.py \
          --model ${model} \
          --split $split \
          --subset $subset
    done
  done
done
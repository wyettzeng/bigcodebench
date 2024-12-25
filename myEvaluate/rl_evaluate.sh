#!/bin/bash

models=(
  "Qwen2.5-Coder-7B-Instruct"
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
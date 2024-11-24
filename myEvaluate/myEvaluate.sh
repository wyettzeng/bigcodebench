#!/bin/bash

models=(
  "Qwen/CodeQwen1.5-7B-Chat"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "NTQAI/Nxcode-CQ-7B-orpo"
  "meta-llama/Meta-Llama-3-8B-Instruct"
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
        # greedy
        python myEvaluate/myEvaluate.py \
          --model ${model} \
          --split $split \
          --subset $subset \
          --n_samples 1 \
          --do_eval False 

        # # generate
        python myEvaluate/myEvaluate.py \
          --model ${model} \
          --split $split \
          --subset $subset \
          --n_samples 16 \
          --do_eval False 

        #eval
        python myEvaluate/myEvaluate.py \
          --model ${model} \
          --split $split \
          --subset $subset \
          --n_samples 16 \
          --do_eval True 
    done
  done
done
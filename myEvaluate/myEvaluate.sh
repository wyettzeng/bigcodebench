#!/bin/bash

models=(
  "codellama/CodeLlama-7b-Instruct-hf"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "meta-llama/Llama-3.1-8B-Instruct"
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

n_lst=(
  16
  32
  64
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

      for n in ${n_lst[@]}
      do
        # # generate
        python myEvaluate/myEvaluate.py \
          --model ${model} \
          --split $split \
          --subset $subset \
          --n_samples ${n} \
          --do_eval False \
          --tp 8

        #eval
        # eval not working on ping
        # python myEvaluate/myEvaluate.py \
        #   --model ${model} \
        #   --split $split \
        #   --subset $subset \
        #   --n_samples ${n} \
        #   --do_eval True 
      done
    done
  done
done
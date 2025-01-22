#!/bin/bash
export BIGCODEBENCH_TIMEOUT_PER_TASK=10
models=(
  # "codellama/CodeLlama-7b-Instruct-hf"
  # "mistralai/Mistral-7B-Instruct-v0.3"
  # "meta-llama/Llama-3.1-8B-Instruct"
  # "Qwen/CodeQwen1.5-7B-Chat"
  # "Qwen/Qwen2.5-Coder-7B-Instruct"
  # "NTQAI/Nxcode-CQ-7B-orpo"
  # "meta-llama/Meta-Llama-3-8B-Instruct"
  # CodeDPO/llama3-RL-both-E2-0117-ckpt1624
  /home/dongfu/.cache/modelscope/hub/jasperhaozhe/coder-RL-both-E2-0117-ckpt1719
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

      python myEvaluate/myEvaluate.py \
        --model ${model} \
        --split $split \
        --subset $subset \
        --n_samples 1 \
        --do_eval True 

      # for n in ${n_lst[@]}
      # do
      #   mkdir -p logs/${model}
      #   # # generate
      #   echo "generating ${model} ${split} ${subset} ${n}"
      #   python myEvaluate/myEvaluate.py \
      #     --model ${model} \
      #     --split $split \
      #     --subset $subset \
      #     --n_samples ${n} \
      #     --do_eval False \
      #     --tp 4 \
      #     --bs 4 > logs/${model}/${split}_${subset}_${n}_gen.txt 2>&1

      #   #eval
      #   # eval not working on ping
      #   echo "evaluating ${model} ${split} ${subset} ${n}"
      #   python myEvaluate/myEvaluate.py \
      #     --model ${model} \
      #     --split $split \
      #     --subset $subset \
      #     --n_samples ${n} \
      #     --do_eval True > logs/${model}/${split}_${subset}_${n}_eval.txt 2>&1 &
      # done
    done
  done
done
wait
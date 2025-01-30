#!/bin/bash
# export BIGCODEBENCH_TIMEOUT_PER_TASK=10

models=(
  "codellama/CodeLlama-7b-Instruct-hf"
  "mistralai/Mistral-7B-Instruct-v0.3"
  # "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  # "Qwen/CodeQwen1.5-7B-Chat"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  # "NTQAI/Nxcode-CQ-7B-orpo"
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

# for model in ${models[@]}
# do
#   for split in ${splits[@]}
#   do
#     for subset in ${subsets[@]}
#     do
#       # greedy
#       # python myEvaluate/myEvaluate.py \
#       #   --model ${model} \
#       #   --split $split \
#       #   --subset $subset \
#       #   --n_samples 1 \
#       #   --do_eval False 

#       for n in ${n_lst[@]}
#       do
#         # # generate
#         # python myEvaluate/myEvaluate.py \
#         #   --model ${model} \
#         #   --split $split \
#         #   --subset $subset \
#         #   --n_samples ${n} \
#         #   --do_eval True \
#         #   --tp 2 \
#         #   --bs 4 \

#         #eval
#         # eval not working on ping
#         python myEvaluate/myEvaluate.py \
#           --model ${model} \
#           --split $split \
#           --subset $subset \
#           --n_samples ${n} \
#           --do_eval True &
#       done
#     done
#   done
# done
# wait


for model in "${models[@]}"
do
  for n in "${n_lst[@]}"
  do
    (
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
            --do_eval True
        done
      done
    ) &
  done
done

wait
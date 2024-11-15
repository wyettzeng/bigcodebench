#!/bin/bash

models=(
  "Qwen/CodeQwen1.5-7B-Chat"
  # "Qwen/Qwen2.5-Coder-7B-Instruct"
  "NTQAI/Nxcode-CQ-7B-orpo"
  "meta-llama/Meta-Llama-3-8B-Instruct"
)

for model in ${models[@]}
do
  bigcodebench.evaluate \
    --model ${model} \
    --split instruct \
    --subset full \
    --backend vllm

  # best of N
  bigcodebench.evaluate \
    --model ${model} \
    --split instruct \
    --subset full \
    --backend vllm \
    --n_samples 16 \
    --temperature 1.0
done
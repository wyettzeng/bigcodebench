#!/bin/bash
set -ex
export HF_ENDPOINT=https://hf-mirror.com
export HOST_IP=0.0.0.0
python=/home/ma-user/.conda/envs/pytorch2.4/bin/python
# $python -m pip install fire
models=(
  ${1}
  # "/home/ma-user/work/haozhe/code/math_workflow/Mammoth-code/aapublic/training_for_codeRM/output_models/r0_lr20_w2_E1_rho2_kl0.5_codeRL_IPS_data0_E1_pre0_w2_ds1"
  # "/home/ma-user/work/share_base_models/Qwen2.5-Coder-7B-Instruct"
  # "Qwen/CodeQwen1.5-7B-Chat"
  # "Qwen/Qwen2.5-Coder-7B-Instruct"
  # "NTQAI/Nxcode-CQ-7B-orpo"
  # "meta-llama/Meta-Llama-3-8B-Instruct"
)

splits=(
  "complete"
  "instruct"
)

subsets=(
  "hard"
  "full"
)

cd /home/ma-user/work/haozhe/code/math_workflow/Mammoth-code/aapublic/evaluation_for_codeRL/bigcodebench

for model in ${models[@]}
do
  for split in ${splits[@]}
  do
    for subset in ${subsets[@]}
    do
        # greedy
        # $python myEvaluate/myEvaluate.py \
        #   --model ${model} \
        #   --split $split \
        #   --subset $subset \
        #   --n_samples 1 \
        #   --do_eval False 

        # # generate
        $python myEvaluate/myEvaluate.py \
          --model ${model} \
          --split $split \
          --subset $subset \
          --n_samples 16 \
          --do_eval False 

        #eval
        $python myEvaluate/myEvaluate.py \
          --model ${model} \
          --split $split \
          --subset $subset \
          --n_samples 16 \
          --do_eval True 
    done
  done
done
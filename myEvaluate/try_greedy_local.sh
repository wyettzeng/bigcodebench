#!/bin/bash
set -ex

export HF_ENDPOINT=https://hf-mirror.com
export HOST_IP=0.0.0.0
# python=/home/ma-user/.conda/envs/pytorch2.4/bin/python
# $python -m pip install fire

cd /home/ma-user/work/haozhe/workspace/aacodeRL/evaluation_for_codeRL/bigcodebench
# sudo mkdir /home/tmp
export TMPDIR=/home/tmp
python -m pip install -e .
python myEvaluate/myEvaluate.py \
          --model ${1} \
          --split ${2} \
          --subset ${3} \
          --n_samples 1 \
          --do_eval False 
# $python -m pip install -e .
# $python myEvaluate/myEvaluate.py \
#   --model ${model} \
#   --split $split \
#   --subset $subset \
#   --n_samples 1 \
#   --do_eval True 
# cd /home/ma-user/work/haozhe/code/math_workflow/Mammoth-code/aacodeRL/evaluation_for_codeRL
# for model in ${models[@]}
# do
#   for split in ${splits[@]}
#   do
#     for subset in ${subsets[@]}
#     do
#         # greedy
#         $python myEvaluate/myEvaluate.py \
#           --model ${model} \
#           --split $split \
#           --subset $subset \
#           --n_samples 1 \
#           --do_eval True 

#         # # # generate
#         # $python myEvaluate/myEvaluate.py \
#         #   --model ${model} \
#         #   --split $split \
#         #   --subset $subset \
#         #   --n_samples 16 \
#         #   --do_eval False 

#         # #eval
#         # $python /home/ma-user/work/haozhe/workspace/aacodeRL/evaluation_for_codeRL/bigcodebench/myEvaluate/myEvaluate.py \
#         #   --model ${model} \
#         #   --split $split \
#         #   --subset $subset \
#         #   --n_samples 1 \
#         #   --do_eval True 
#     done
#   done
# done
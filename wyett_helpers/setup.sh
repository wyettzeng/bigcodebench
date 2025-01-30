#!/bin/bash

conda create -n bigcodebench python=3.10
conda init
conda activate bigcodebench

# By default, you will use the remote evaluation API to execute the output samples.
pip install bigcodebench --upgrade

# You are suggested to use `flash-attn` for generating code samples.
pip install packaging ninja
pip install flash-attn --no-build-isolation
# Note: if you have installation problem, consider using pre-built
# wheels from https://github.com/Dao-AILab/flash-attention/releases

pip install -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt
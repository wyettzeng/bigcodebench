# Install
pip install transformers

# Overview
The RL pipeline has two steps:
1. prepare a dataset, the reward are preprocessed 
2. obtaining logps for the query-response 
3. training with the dataset (providing rewards) and the logp

# Steps
## 1. Preparing the dataset
required keys:
- uid: unique ids for query-response 
- prompt_pretokenized

optional keys:
- reward key: such as `normalized_reward`
- output key: such as `unprocessed_response` 

## 2. Obtaining the logps
Simply `bash run_logp.sh {datapath} {modelpath} {tag}`

```bash
export fp="${1}" # the parquet path
export logp_path="None"
export lr=10 # 10e-6
export gpu=8
export memo="_${3}_scaleonly_r0_logp" 
export mp=${2} # the model path
export bsz=4 # batchsize
```

Note: the logp will be output to the folder `${fp}${memo}`, in the pickle format. 

## 3. Training 
Simply `bash run_logp.sh {datapath} {modelpath} {tag} {rewardkey} {rspkey} {datafilter}`

- setting up the wandb
    - in `train.py`, replace `wandb_key` with your own.
    - set up the project name in the bash script. (see below)

- setting up deepspeed env variables
```bash
--master_addr ${MASTER_ADDR} \
--nproc_per_node=${NUM_GPUS} \
--master_port=${MASTER_PORT} \
--nnodes=${WORKER_NUM} \
--master_addr=${MASTER_ADDR} \
--node_rank=${NODE_RANK} \
```

- setting up the params
```bash
export fp="${1}" # the parquet data path
export modelname=${3} # the model tag
export logp_path=${1}_${3}_scaleonly_r0_logp # the logp folder
export bsz=1 # batchsize per card
export gacc=16 # gradient accumulation steps
export lr=10
export gpu=8
export kl=0.5
export ml=2000 # output max length
export proj="CodeRLTrain" # the wandb project name
export rkey=${4}
export rspkey=${5}
export datafilter=${6}
export memo="${proj}_${modelname}_IPS_max${ml}_${3}_${4}_${5}_${6}"
export numepoch=3
export mp=${2} # the model path 
```

Here `rkey` is the key name you use for reward, `rspkey` is the key name you use for response, `datafilter=none` will do RL training, `datafilter=posonly` will filter `reward>0.75`. 
set -ex
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
out_base=/home/ma-user/work/haozhe/workspace/aacodeRL/training_for_codeRM/output_models

logp_path=${logp_path:-'None'}
numepoch=${numepoch:-'1'}
lr=${lr:-'20'}
logp=${logp:-'0'}
kl=${kl:-'1'}
memo=${memo:-''}
proj=${proj:-"CodeRL"}
rkey=${rkey:-"accuracy"}
rspkey=${rspkey:-"unprocessed_response"}
datafilter=${datafilter:-"None"}
tag="lr${lr}_w${isweighted}_E${numepoch}_kl${kl}_${memo}"

maxlength=1400
ml=${ml:-"${maxlength}"}

modelname="${tag}_E${numepoch}"
export MODEL_PATH=${mp}
export OUTPUT_PATH="${out_base}/${modelname}"

per_device_train_batch_size=1
gradient_accumulation_steps=16

bsz=${bsz:-"${per_device_train_batch_size}"}
gacc=${gacc:-"${gradient_accumulation_steps}"}

################## env variables for deepspeed
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export MASTER_ADDR=localhost
export MASTER_PORT="6066"
export WANDB_PROJECT=${proj}
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1 
export OMP_NUM_THREADS=128

NODE_RANK=0
export NUM_GPUS=${gpu}
export WORKER_NUM=1
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3930
MASTER_PORT="6066"
flash_attn=False
# torchrun=/home/ma-user/.conda/envs/pytorch2.4/bin/torchrun

env="dlc"

if [ $NUM_GPUS -gt 1 ]; then
echo "dlc"
export NCCL_NET=IB
export NCCL_NET_PLUGIN=none
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=15

MASTER_HOST="$VC_WORKER_HOSTS"
IFS=',' read -r -a array <<< "${VC_WORKER_HOSTS}"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
WORKER_NUM="$MA_NUM_HOSTS"
NODE_RANK="$VC_TASK_INDEX"
NUM_GPUS="$MA_NUM_GPUS"
flash_attn=False
fi
####################

echo -e "MASTER_HOST:$MASTER_HOST\nMASTER_ADDR:$MASTER_ADDR\nNODE_RANK:$NODE_RANK\nWORKER_NUM:$WORKER_NUM\n"
echo -e "NUM_GPUS:$NUM_GPUS\nMODEL_PATH:$MODEL_PATH\nper_device_train_batch_size=$bsz\ngradient_accumulation_steps=$gacc" 
torchrun --master_addr ${MASTER_ADDR} \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=${MASTER_PORT} \
  --nnodes=${WORKER_NUM} \
  --master_addr=${MASTER_ADDR} \
  --node_rank=${NODE_RANK} \
  train.py \
    --model_name_or_path ${MODEL_PATH} \
    --model_max_length ${ml} \
    --run_name ${modelname} \
    --memo ${memo} \
    --data_path ${fp} \
    --data_config ${fp} \
    --logp_path ${logp_path} \
    --bf16 True \
    --output_dir ${OUTPUT_PATH} \
    --num_train_epochs 1 \
    --dataset_repeat ${numepoch} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --gradient_accumulation_steps ${gacc} \
    --evaluation_strategy "steps" \
    --eval_steps "40" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 6 \
    --learning_rate "${lr}e-6" \
    --do_logp ${logp} \
    --kl_weight ${kl} \
    --end_lr 6e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --save_safetensors True \
    --flash_attn ${flash_attn} \
    --logging_steps 1 \
    --rkey ${rkey} \
    --rspkey ${rspkey} \
    --datafilter ${datafilter} \
    --fsdp "full_shard auto_wrap" 
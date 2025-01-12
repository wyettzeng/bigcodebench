
export fp="${1}"
export modelname=${3}
export logp_path=${1}_${3}_scaleonly_r0_logp #"${fp}_${modelname}_r${round}_logp"
export bsz=1
export gacc=16
export lr=10
export gpu=8
export kl=0.5
export ml=2000
export proj="CodeRLTrain"
export rkey=${4}
export rspkey=${5}
export datafilter=${6}
export memo="${proj}_${modelname}_IPS_max${ml}_${3}_${4}_${5}_${6}"
export numepoch=3
export mp=${2}

# export mp="/home/ma-user/work/haozhe/code/math_workflow/Mammoth-code/aaout/dsmath_sft"
# base_model_path="/home/ma-user/work/share_base_models/AI-ModelScope/Mistral-7B-v0___1/"


bash base.sh
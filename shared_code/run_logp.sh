
export fp="${1}"
export logp_path="None"
export lr=10
export gpu=8
export kl=0.5
export memo="_${3}_scaleonly_r0_logp"
export numepoch=1
export mp=${2}
export bsz=4
export logp=1 # logp mode

bash base.sh
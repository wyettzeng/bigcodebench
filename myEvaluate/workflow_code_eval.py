import sys
import copy
import os
from glob import glob
sys.path.insert(
    0,
    "/home/ma-user/work/haozhe/code/math_workflow/Mammoth-code/aatools/megv3_workflow"
)
from wf_utils import release_workflow, job, one_card, eight_card, a100pool_id, xxpool_id, prodpool_id

def get_cmd(version):
    return "bash /home/ma-user/work/haozhe/workspace/aacodeRL/evaluation_for_codeRL/bigcodebench/myEvaluate/try_greedy_local.sh {model} {ver1} {ver2}"
    
base_config = [
    "eval_{ver1}_{ver2}_{modelname}",
    None,
    "inftech/math:1.22_code_eval",
    one_card,
    1,
]
priority=3
modelname = 'qwen_acc'
modelpath = "/home/ma-user/work/share_base_models/Qwen2.5-Coder-7B-Instruct"
ml = 1000
dataid=0
pool_id = prodpool_id
basefolder = "/home/ma-user/work/haozhe/workspace/aacodeRL/training_for_codeRM"

savepath = "/home/ma-user/work/haozhe/workspace/aacodeRL/training_for_codeRM/output_models/r0_lr5_w2_E0.05_rho2_kl0.5_CodeRL_qwen_acc_v2_IPS_max1000_v2_data0_E0.05_pre0_w2_ds1/"
# savepath = f"{basefolder}/output_models/r0_lr5_w2_E0.1_rho2_kl0.5_codeRL_qwen_acc_IPS_max1000_data0_E0.1_pre0_w2_ds1"
# models = [
#     modelpath,
#     savepath
# ]
# models = glob(savepath+'*')
models = glob(savepath+'*')
name = modelname.replace("_", "").replace(".","")
# print(name)
# return
logp_config = [
    "logp_{name}_r{round}",
    "bash {basefolder}/debug_logp.sh {dataid} {mp} {modelname} {dataid}",
    "inftech/math:1.20",
    eight_card,
    1,
]

train_config = [
    "train_{name}_r{round}",
    "bash {basefolder}/debug_train.sh {dataid} {mp} {modelname} {dataid}",
    "inftech/math:1.20",
    eight_card,
    1,
]



# models = ['qwen25_1209_7b_7_000','qwen25_1209_7b_7_012', 'qwen25_1209_7b_7_025', 'qwen25_1209_7b_7_050']
print(models)

name = models[-1]
num_total = 2

alljobs = []

jname = logp_config[0]
cmd = logp_config[1]
logp_config[0] = jname.format(name=modelname, round=dataid)
logp_config[1] = cmd.format(dataid=dataid, mp=modelpath, modelname=modelname, basefolder=basefolder)
logp = job(*logp_config, dep=[], priority=priority, pool_id=pool_id)
# alljobs.append(logp)
jname = train_config[0]
cmd = train_config[1]
train_config[0] = jname.format(name=modelname, round=dataid)
train_config[1] = cmd.format(dataid=dataid, mp=modelpath, modelname=modelname, basefolder=basefolder)

train = job(*train_config, dep=[logp], priority=priority, pool_id=pool_id)

# alljobs.append(train)
jobid = 0

for ver1 in ['complete','instruct']:
    for ver2 in ['hard','full']:
        
        for model in models: # ['gsm-hard', 'svamp', 'mawps', 'asdiv']:# ['gsm8k','math']:
            # templ = get_cmd('machine')
            # modelname = model.split('/')[-1]
            templ = get_cmd("")
            cmd = templ.format(model=model, ver1=ver1, ver2=ver2, basefolder=basefolder)
            modelname = model.split(os.path.sep)[-1]
            config = copy.copy(base_config)
            config[1] = cmd
            jobname_templ = config[0]
            config[0] = jobname_templ.format(modelname=modelname, ver1=ver1, ver2=ver2).replace('.','').replace("_","")[:64]
            print(config)
            alljobs.append(job(*config, dep=[], priority=3, pool_id=pool_id))
            jobid += 1
        



fname = f'coder12eval_{modelname.replace(".","").replace("_","")}'
release_workflow(fname[:64], alljobs)
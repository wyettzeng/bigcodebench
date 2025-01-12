#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import os
import json
import math
import datasets
import pdb
from glob import glob
import torch
import numpy as np
import transformers
import pickle as pkl
from torch.utils.data import Dataset
from transformers import Trainer, get_scheduler
import pathlib
# import utils
import re
import random
from tqdm import tqdm 
import torch.distributed as dist
import wandb
from collections import deque, defaultdict
import sys
import pandas as pd
from transformers import TrainerCallback

def print_on_rank_0(*args):
    if dist.get_rank() == 0:
        print(*args)
        
#from trl import SFTTrainer
#os.system("echo $PYTORCH_CUDA_ALLOC_CONF")
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|reserved_special_token_250|>" # for llama only
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>" # "<|reserved_special_token_249|>"
DEFAULT_Pause_TOKEN = "<pause>"
padding_length = 160


class RewardQueue:
    def __init__(self, windowsize=50, batchsize=1, use_running=False, report_num=32): # batchsize=4x8
        self.maxsize = report_num
        self.batchsize = batchsize
        self.q = deque(maxlen=self.maxsize)
        self.init_mean = 0.
        self.std = 1.0
        self.use_running = use_running
        self.report_num = report_num
        # accelerator.print(f"reward queue use_running={use_running} when normalizing rewards.")
    
    def append(self, entry):
        self.q.append(entry)
        
    def extend(self, items):
        self.q.extend(items)
        
    def stats(self, running=False): # running mean seems to make policy worse
        if len(self.q)>self.report_num//2 and running: 
            return np.mean(self.q), self.std 
        else: return None, self.std

    def normalize_reward(self, r):
        m,s = self.stats(self.use_running)
        return (r-m)/s
    
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    padding_side: Optional[str] = field(default="right")
    deepseek_templ: int = field(default=0)
    round_id: int = field(default=0)
    do_logp: int = field(default=0)
    show_kl: int = field(default=1)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_config: str = field(default="")
    template_variation: bool = field(
        default=True, metadata={"help": "whether to use template variation"})
    direct: bool = field(default=False)
    add_pause_to_answer: bool = field(default=False)
    pause: bool = field(default=False)
    is_pretrain: int = field(default=0)
    is_weighted_loss: int = field(default=0)
    logp_path: str = field(default="None", metadata={"help": "Path to the logp data."})
    rw_path: str = field(default="None", metadata={"help": "Path to the reward data."})
    dataset_repeat: float = field(default=1)
    memo: str = field(default="")
    code_mode: int = field(default=0)
    rkey: str = field(default="")
    rspkey: str = field(default="")
    datafilter: str = field(default="")
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    flash_attn: bool = field(default=False)
    run_name: str 
    end_lr: float = field(default=6e-6)
    num_cosine_steps: int = field(default=6000)
    num_warmup_steps: int = field(default=200)
    kl_weight: float = field(default=1.0)
    kl_discount: float = field(default=0.5)
    use_rho: int = field(default=0)
    


transformers.logging.set_verbosity_info()
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    is_llama=False
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    print_on_rank_0("token embedding resized", len(tokenizer))
    if is_llama:
        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    # sources: Sequence[str],
    # targets: Sequence[str],
    data, 
    tokenizer: transformers.PreTrainedTokenizer,
    is_pretrain=False
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources, targets, logps = data
    examples = [s + t for s, t in zip(sources, targets)]
    # examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


def get_code_data(data_arg, tokenizer: transformers.PreTrainedTokenizer, template_variation: bool, ratio=0):
    """ input data paths, output data
    1. load logp
    2. load data 
    3. do shuffling and repeats
    4. construct the dataset
    """
    print_on_rank_0("using get_code_data")
    config, data_path = data_arg
    if os.path.exists(config) and 'config' in config:
        data_config = json.load(open(config))
    else: data_config = dict(default=data_path)
    logp_path = data_args.logp_path
    # rw_path = data_args.rw_path
    isqwen = 'qwen' in data_args.memo 
    filter_key = 'qwen_coder_2.5' if isqwen else 'llama3_instruct'
    ################## 1. loading logp
    logp_dict = dict()
    if logp_path=="None":logp_data_files = glob(data_path+'_logp/*')
    else: logp_data_files = glob(logp_path+'/*')
    for fp in logp_data_files:
        print(fp)
        tmpd = pkl.load(open(fp,"rb"))
        logp_dict.update(tmpd)
    print_on_rank_0("logp num", len(logp_dict))
    ######################################
    all_dat = []
    rlist = []
    #########################
    # what keys will be used for rewards and responses 
    #######################
    rkey = data_args.rkey 
    respkey = data_args.rspkey
    data_filter = data_args.datafilter
    ############# 2. loading data
    for k, data_path in data_config.items():
        paths = [data_path]
        logging.warning(f"{k}: Loading data from {paths}")
        
        alldfs = []
        for dp in tqdm(paths):
            print(dp)
            df = pd.read_parquet(dp)
            alldfs.append(df)
            
            if 'uid' not in df:
                df['uid'] = df['qid'] + '-' + df['sample_id'].astype(str)
            df['response'] = df[respkey]
        alldfs = pd.concat(alldfs).reset_index()
        
        df = alldfs
        df['whitened_reward'] = df[rkey]
        
        if data_filter == 'posonly':
            df = df[df[rkey]>0.75] 
        isna = df['whitened_reward'].isna() 
        print('nan values', isna.sum())
        df[rkey] = [np.clip(x, -1.5, 1.5) for x in df[rkey]]
        newdf = df
        all_dat.extend(np.arange(len(newdf)))
    print(f"unique ids {len(df.uid.unique())}, data num {len(df)}")
    if len(df.uid.unique())!=len(df):
        print_on_rank_0('++++++++++++++++')
        print_on_rank_0('warning: unique id does no align')
        print_on_rank_0('++++++++++++++++')
        print_on_rank_0(df.uid[0])
        raise Exception('warning: unique id does no align')
    
    repeat = data_args.dataset_repeat-1
    ########## 3. shuffle and repeat
    # to handle data repeat, we repeat the indexes instead of directly repeating the data
    ##########################
    random.seed(42)
    random.shuffle(all_dat)
    new_dat = []
    if repeat>0: 
        for _ in range(int(repeat)):
            new_dat.extend(all_dat)
        all_dat.extend(new_dat)
    elif repeat<0: 
        num_keep = int(data_args.dataset_repeat*len(all_dat))
        all_dat = all_dat[:num_keep]
    
    random.seed(42)
    random.shuffle(all_dat)
    all_dat = all_dat[:1000]
    logging.warning("Formatting inputs...")
        
    sources = []
    targets = []
    weights = []
    logps = []
    ids = []
    estimated = []
    outcomes = []
    typeset = set()
    
    ######### print stats of the rewards 
    rlist = newdf['whitened_reward'].values
    if len(rlist)>0: 
        rmean = np.mean(rlist)
        rstd = np.std(rlist)
        print('estimated reward stats', rmean, rstd, np.max(rlist), np.min(rlist))
    
    ######### 4. constructing the data for dataloader
    # source: input to LM
    # target: output that will compute loss
    # weights: rewards. use this name due to legacy
    # logp: reference logp
    # ids: qa sample id
    # estimated, outcomes: they exist due to legacy reasons
    ##########################
    for idx in tqdm(all_dat):
        example = newdf.iloc[idx]
        messages = json.loads(example['prompt_pretokenized']) # user: xxx assistant: xxx 
        source = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True
        )
        prefix = messages[1]
       
        source = source + prefix['content']
        target = example['response'].strip('python').strip('```')+tokenizer.eos_token
        
        sources.append(source)
        targets.append(target)
        
        weights.append(example['whitened_reward'])
        qid = example['uid']
        
        logp = logp_dict.get(qid, None) # if not is_syn else np.array([0.0])
        if logp is None and model_args.do_logp==0: continue # training mode with logp=None, skip
        logps.append(logp)
        ids.append(example['uid'])
        estimated.append(None)
        outcomes.append(1.0 if example['whitened_reward']>0 else 0)
    
    print_on_rank_0("before skipping", len(all_dat))
    print_on_rank_0("after skipping", len(outcomes))
    if len(outcomes)==0:
        print_on_rank_0('++++++++++++++++')
        print_on_rank_0('warning: there is no logp')
        print_on_rank_0('++++++++++++++++')
        print_on_rank_0('data uid', qid)
        raise Exception("there is no logp")
        print_on_rank_0('logp uid', list(logp_dict.keys())[0])
    print_on_rank_0("rewards mean", np.mean(weights))
    print_on_rank_0("rewards max min", np.max(weights), np.min(weights))
  
    num_trn = len(sources)
    
    ratio = int(0.99 * len(sources))
    if model_args.do_logp==0: num_trn = ratio 
    
    print_on_rank_0('='*20, 'peek data')
    print_on_rank_0(sources[0])
    print_on_rank_0("="*20)
    print_on_rank_0(targets[0])
    train_data = [sources[0:num_trn],targets[0:num_trn], weights[:num_trn], logps[:num_trn], ids[:num_trn], estimated[:num_trn], outcomes[:num_trn]]
    print_on_rank_0([len(x) for x in train_data])
    eval_data = [sources[num_trn:],targets[num_trn:], weights[num_trn:], logps[num_trn:], ids[num_trn:], estimated[num_trn:], outcomes[num_trn:]]
    del newdf
    return train_data,eval_data


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data):
        super(SupervisedDataset, self).__init__()
        
        self.sources = data[0]
        self.targets = data[1]
        self.weights = data[2]
        self.logps = data[3]
        self.ids = data[4]
        self.estimated = data[5]
        self.outcomes = data[6]
        print('supervised dataset', len(self.sources))

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], weights=self.weights[i])

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=(self.targets[i],self.weights[i], self.logps[i], self.ids[i], self.estimated[i], self.outcomes[i]))

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    is_pretrain: bool
    is_weighted_loss: bool

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, weights = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "weights"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            weights=weights, 
        )
        # if self.is_weighted_loss:
        #     ret['weights'] = weights
        return ret 
        

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        weights = []
        logps = []
        ids = []
        rewards = []
        outcomes = []
        # use_estimated_reward = False
        for instance in instances:
            source = instance['input_ids']
            target,weight,logp,qid,rew,oc = instance['labels']
            
            sources.append(source)
            targets.append(target)
            weights.append(weight)
            logps.append(logp)
            ids.append(qid)
            outcomes.append(oc)
            # if rew is array, append 0 for end token
            # if use_estimated_reward:
            #     rewards.append(None if rew is None else np.append(rew,1.0))

        data_dict = preprocess((sources, targets, logps), self.tokenizer, self.is_pretrain)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            # weights=weights,
            # logps=logps, 
            # ids=ids
        )
        
        # if self.is_weighted_loss>=1 or model_args.round_id>=1:
        ret['outcomes'] = outcomes
        ret['weights'] = weights 
        ret['logps'] = logps # torch.nn.utils.rnn.pad_sequence([torch.from_numpy(x[0]).to(device=labels.device) for x in logps], batch_first=True, padding_value=0.)
        ret['ids'] = ids 
        return ret 

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_func = get_code_data
    train_data,eval_data = data_func(tokenizer=tokenizer, data_arg=(data_args.data_config, data_args.data_path),
                                      template_variation=data_args.template_variation)
    train_dataset,eval_dataset = SupervisedDataset(train_data),SupervisedDataset(eval_data)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, is_pretrain=data_args.is_pretrain==1, is_weighted_loss=True)
    if model_args.do_logp==1:
        return dict(
            # train_dataset=train_dataset, 
            eval_dataset=train_dataset, 
            data_collator=data_collator)
    else: 
        return dict(train_dataset=train_dataset, 
                    eval_dataset=eval_dataset, 
                    data_collator=data_collator)


log_keys = []
for prefix in ['pos','neg']:
    for suffix in ['kl','rho','logp','loss','reward','count']:
        log_keys.append(prefix+'_'+suffix)
        if suffix not in ['kl','rho','count']:
            log_keys.append('syndata_'+prefix+'_'+suffix)
log_keys.append('syndata_count')

class CustomWandbCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        # Custom tensors to log, assuming you have them computed somewhere in your training loop
        # For example:
        
        super().on_log(args, state, control, **kwargs)
        for k in log_keys:
            custom_tensor_1 = kwargs.get(k, None)
            if custom_tensor_1 is not None:
                wandb.log({f"stats/{k}": custom_tensor_1})
            
            
class WeightedLossTrainer(Trainer):
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_probs = dict()  # Initialize log_probs here
        self.log_keys = log_keys
        self.log_freq = 8*10 
        batchsize = 2 
        self.niter = 0
        self.log_queue = {k:RewardQueue(batchsize=self.log_freq*batchsize, report_num=self.log_freq*batchsize) for k in self.log_keys}

    def compute_loss(self, model, inputs, return_outputs=False):
        """ given model and inputs, compute reinforce loss
        1. token loss
        2. KL term
        3. off-policy correction term 
        4. loss = token_loss * reward * OPC + KL
        """
        self.niter += 1
        weights = inputs.pop("weights", None)  # Extract and remove the weights from inputs
        token_reflogps = inputs.pop("logps", None)
        outcomes = inputs.pop("outcomes", None)
        rho_version = training_args.use_rho
        has_reflogp = True
        # import pdb; pdb.set_trace()
        if model_args.do_logp==1: 
            has_reflogp = False 
        else: 
            # list of float, the per-token reference logp for each sample
            sample_reflogps = [None if x is None else np.mean(x) for x in token_reflogps]
            
        
        ids = inputs.pop("ids", None)
        num_sample = len(ids)
        if model_args.do_logp==1:
            with torch.no_grad():
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        labels = inputs.get("labels")
        logits = outputs.get("logits")
        
        rewards_ = torch.tensor(weights, dtype=torch.float32).to(logits.device)
            
        use_estimated_reward = False
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_mask = (shift_labels!=IGNORE_INDEX).float().to(device=shift_logits.device)
        
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        
        mask = loss_mask.view(num_sample, -1)
        
        tmp = self.loss_fct(shift_logits, shift_labels)
        
        tmp = tmp.view(num_sample, -1)
        
        if model_args.do_logp==1:
            for uid, smask,logp in zip(ids, loss_mask, -tmp):
                final = torch.masked_select(logp, smask==1).detach().cpu().numpy()
                self.log_probs[uid] = final 
        
        ################ compute loss
        # 1. denominator: effective num of tokens to compute loss 
        # 2. token_loss: -logp
        # 3. KL term and off-policy correction
        # 4. loss = token_loss * OPC * reward + KL
        ##################################
        denom = mask.sum(-1)
        denom[denom==0.] = 1e-6
        token_loss = tmp * mask
        sample_loss = token_loss.sum(-1)/denom
        sample_logp = -sample_loss
        token_logp = -tmp 
        sample_rewards = rewards_.view(num_sample, 1)
        #########
        # handle KL term and off-policy correction term
        #############################################
        if has_reflogp:
            ########### sample-wise KL 
            sample_estimated_kl = [0.0 if b is None else b-a for a,b in zip(sample_logp,sample_reflogps)]
            
            ########### sample-wise off-policy correction term
            sample_rhos = torch.ones_like(token_logp)
            
            for i, (m, tlogp) in enumerate(zip(mask, token_reflogps)):
                # if isself:
                tlogp = torch.from_numpy(tlogp.astype(np.float32)).to(token_logp.device)
                a = token_logp[i][m==1]
                b = tlogp[:len(a)] # 如果被truncate b lenth> a lenth
                sample_rhos[i][m==1] = torch.exp(a-b).detach()
                    
            ##### IPS clipping, ref: PPO
            sample_clipped_rhos = torch.clamp(sample_rhos, max=1.2)
            sample_rhos_display = (sample_clipped_rhos * mask).sum(-1)/(mask.sum(-1)+1e-6) # for display purpose
            
            if len(sample_clipped_rhos.shape)==1:
                sample_clipped_rhos =  sample_clipped_rhos.view(num_sample, 1)
        
        token_rho_loss = token_loss * (sample_clipped_rhos if has_reflogp else 1.0)
        token_rho_loss = token_rho_loss * sample_rewards
        sample_rho_loss = token_rho_loss.sum(-1) / denom
        loss = sample_rho_loss.mean()
        
        kl_penalty = 0.0*loss
        
        # log 
        kl_pos_weight = training_args.kl_weight
        kl_neg_weight = kl_pos_weight * training_args.kl_discount
        
        loss = loss + kl_penalty
        
        ########### stats for RL training, for display purpose
        if has_reflogp:
            for ii, (r, mm, v, lp, rh, b, ls) in enumerate(zip(outcomes, mask, sample_estimated_kl, sample_logp, sample_rhos_display, token_reflogps, sample_rho_loss)):
                ver = 'neg' if r<0.5 else 'pos'
                if token_reflogps[ii] is None: 
                    continue

                if ver=='pos': kl_penalty += v * kl_pos_weight
                elif ver=='neg': kl_penalty += v * kl_neg_weight
                
                if b is not None:
                    name = f"{ver}_kl"
                    self.log_queue[name].append(v.item())
                
                name = f"{ver}_rho"
                self.log_queue[name].append(torch.mean(rh).item())  
                
        for ii, (r, mm, lp, ls, sr) in enumerate(zip(outcomes, mask, sample_logp, sample_rho_loss, sample_rewards)):
            ver = 'neg' if r<0.5 else 'pos'
            if token_reflogps[ii] is None: 
                prefix = 'syndata_'
                ver = prefix + ver
            
            
            name = f"{ver}_logp"
            self.log_queue[name].append(lp.item()*abs(sr.item()))   
            name = f"{ver}_loss"
            self.log_queue[name].append(ls.item())   
            
            if use_estimated_reward:
                entry_reward = (sample_rewards[ii] * mm).sum()/(mm.sum())
            else: entry_reward = sample_rewards[ii]
            name = f"{ver}_reward"
            self.log_queue[name].append(entry_reward.item())   
            for k in ['syndata','pos','neg']:
                name = f"{k}_count"
                self.log_queue[name].append(float(ver.startswith(k)))   
                
        
        
        if self.accelerator.is_main_process and self.niter%self.log_freq==0:
            log_values = dict()
            for k in self.log_keys:
                tmp = self.log_queue[k].stats(running=True)[0]
                if tmp is None: continue 
                log_values[k] = tmp 
            self.callback_handler.on_log(self.args, self.state, self.control, logs=log_values)
            
        if return_outputs: 
            return (loss, outputs)
        else: return loss

wandb_key = "0a7c185570b683512cc61d2209e91a952eee0ad9"
def train():
    wandb.login(relogin=True, key=wandb_key)
    print_on_rank_0('Start Loading Model')
    print_on_rank_0(training_args)

    if training_args.flash_attn:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            use_cache=True,
        ).to('cuda')
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            #attn_implementation='eager'
        ).to('cuda')
    print_on_rank_0(model)
    print_on_rank_0('Start building tokenizer')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast = True,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        truncation_side='right'
        # use_fast=False, 
    )
    # pdb.set_trace()
    print_on_rank_0("*"*50)
    print_on_rank_0("Before adding, tokenizer length: ",len(tokenizer))
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = "<|reserved_special_token_249|>" if "llama" in model_args.model_name_or_path.lower() else DEFAULT_UNK_TOKEN 

    print('====== special tokens')
    print(special_tokens_dict)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
        is_llama="llama" in model_args.model_name_or_path.lower()
    )
    print_on_rank_0("*"*50)
    print_on_rank_0("After adding, tokenizer length: ",len(tokenizer))

    print_on_rank_0('Start building data module')
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    print_on_rank_0('Start building the trainer module')
    callbacks = []
    if model_args.do_logp==1:
        # assert data_args.data_path.endswith('.pkl'), f"data_path {data_args.data_path} must be pickle"
        savefolder = data_args.data_path+data_args.memo
        if dist.get_rank()==0:
            os.makedirs(savefolder, exist_ok=True)
        TrainerClass = WeightedLossTrainer
    else:
        TrainerClass = WeightedLossTrainer
        callbacks = [CustomWandbCallback()]
        print_on_rank_0('callbacks added:', callbacks)
    
    print_on_rank_0('using trainer', str(TrainerClass))
    trainer = TrainerClass(model=model, 
                           tokenizer=tokenizer, 
                           args=training_args, 
                           callbacks=callbacks,
                           **data_module)

    if model_args.do_logp==1:
        eval_result = trainer.evaluate()
        # path = data_args.data_path
        savepath = savefolder+f'/{dist.get_rank()}.pkl'
        pkl.dump(trainer.log_probs, open(savepath,"wb"))
        print("writing", savepath, len(trainer.log_probs))
        
        exit(0)
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # print_on_rank_0("pretrain")
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

def get_cosine_with_end_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, end_learning_rate=0.0):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * num_cycles * progress))
        return cosine_decay * (1 - end_learning_rate) + end_learning_rate

    return get_scheduler(lr_lambda, optimizer, num_warmup_steps, num_training_steps)

if __name__ == "__main__":
    train()

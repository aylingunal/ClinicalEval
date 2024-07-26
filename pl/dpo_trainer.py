
import torch
import math
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from accelerate import Accelerator
from typing import Optional
import torch.nn.functional as F
import tqdm

class DPOTrainer():
    def __init__(self,dpo_config,data_dict):
        self.dpo_config = dpo_config
        self.policies, self.refs, self.tokenizers = [], [], []
        self.train_data = data_dict['train']
        self.eval_data = data_dict['eval']
      #  self.accelerator = Accelerator()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load models (policies) and maintain original copies (ref)
    def load_models(self,):
        # don't re-load if you don't need to!!
        if len(self.policies) == 0 and len(self.refs) == 0 and len(self.tokenizers) == 0:
            for model_name in self.dpo_config.model_names:
                pol_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                                 device_map=self.device)
                self.policies.append(pol_model)
                ref_model = AutoModelForCausalLM.from_pretrained(model_name,device_map=self.device)
                self.refs.append(ref_model)
            # load tokenizers
            for tokenizer_name in self.dpo_config.model_names:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,device_map=self.device,quantization_config=bnb_config)
                self.tokenizers.append(tokenizer)
        # move stuff over to accelerator TODO

    # full train
    def _train_individual(self,model_idx):
        # load model
        policy = self.policies[model_idx]
        ref = self.refs[model_idx]
        tokenizer = self.tokenizers[model_idx]
        # set optimizer
        optimizer = torch.optim.AdamW(policy.parameters(),lr=self.dpo_config.lr)
        # model in training mode
        policy.train()
        bs = self.dpo_config.batch_size
        # set up training
        for epoch in range(self.dpo_config.num_epochs):
            for batch_ in self.train_data:
                # grab batch items; batch is like {transcript: [], note0: [], ...}
                str_queries = batch_['transcript']
                str_note0s = batch_['note0']
                str_note1s = batch_['note1']
                str_note2s = batch_['note2']
                # tokenize
                ids_queries = tokenizer(str_queries,return_tensors='pt').input_ids
                ids_note0s = tokenizer(str_note0s,return_tensors='pt').input_ids
                ids_note1s = tokenizer(str_note1s,return_tensors='pt').input_ids
                ids_note2s = tokenizer(str_note2s,return_tensors='pt').input_ids
                print('ids queries: ',ids_queries)
                print('ids notes0: ',ids_note0s)
                # concatenate query and response tensors
                tensors_note0 = torch.cat((ids_queries,ids_note0s),dim=-1)
                tensors_note1 = torch.cat((ids_queries,ids_note1s),dim=-1)
                tensors_note2 = torch.cat((ids_queries,ids_note2s),dim=-1)
                # forward pass; attend all tokens
                logits_note0, _, _ = policy(**{"input_ids":tensors_note0,"attention_mask":torch.ones_like(tensors_note0)})
                logits_note1, _, _ = policy(**{"input_ids":tensors_note1,"attention_mask":torch.ones_like(tensors_note1)})
                logits_note2, _, _ = policy(**{"input_ids":tensors_note2,"attention_mask":torch.ones_like(tensors_note2)})
                with torch.no_grad():
                    ref_logits_note0, _, _ = ref(**{"input_ids":tensors_note0,"attention_mask":torch.ones_like(tensors_note0)})
                    ref_logits_note1, _, _ = ref(**{"input_ids":tensors_note0,"attention_mask":torch.ones_like(tensors_note1)})
                    ref_logits_note2, _, _ = ref(**{"input_ids":tensors_note0,"attention_mask":torch.ones_like(tensors_note2)})

                policy_logits = (logits_note0,logits_note1,logits_note2)
                ref_logits = (ref_logits_note0,ref_logits_note1,ref_logits_note2)
                policy_logprobs = process_logits(policy_logits)
                ref_logprobs = process_logits(ref_logits)

                loss = individual_loss_dpo(policy_logprobs,ref_logprobs,)

                self.optimizer.zero_grad()
                policy.backward(loss)
                self.optimizer.step()

'''
logprobs are processed before this function (i.e. logits -> logprobs -> single scalar)
params:
{}_logprobs = list of tensors, tensor at index i corresponds to aggregate logprobs for the ith ranked note (so should be a single scalar value)
pref_ranking = dictionary mapping ranks to notes (so {1:note0, ...})
'''
def individual_loss_dpo(policy_logprobs,ref_logprobs,pref_ranking,beta):
    loss = 1.0
    # order notes by rank
    pref_ranking = sorted(pref_ranking, key=lambda x: int(x), reverse=False)
    for k, rank in enumerate(pref_ranking):
        numerator = math.exp(beta * (policy_logprobs[k] - ref_logprobs[k]))
        denominator = 0.0
        for j in range(k,len(pref_ranking)):
            denominator += math.exp(beta * (policy_logprobs[j] - ref_logprobs[j]))
        loss *= (numerator / denominator)
    return loss

def process_logits(logits):
    # aggregate logits and convert to logprobs
    aggr_logits = torch.sum(logits,dim=-1)
    log_probs = F.log_softmax(logits, dim=2)
    return log_probs


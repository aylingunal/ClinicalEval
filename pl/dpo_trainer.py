
import torch
import math
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional
import torch.nn.functional as F
import tqdm
import sys
from accelerate import Accelerator

import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
import subprocess

class DPOTrainer():
    def __init__(self,dpo_config,data_dict):
        self.dpo_config = dpo_config
        self.policies, self.refs, self.tokenizers = [], [], []
        self.accelerator = Accelerator()
        #self.device = self.accelerator.device #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_data = data_dict['train']
        self.eval_data = data_dict['eval']

    # load models (policies) and maintain original copies (ref)
    def load_models(self,):
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        subprocess.run(['nvidia-smi'])
        # don't re-load if you don't need to!
        if len(self.policies) == 0 and len(self.refs) == 0 and len(self.tokenizers) == 0:
            for model_name in self.dpo_config.model_names:
                self.policies.append(AutoModelForCausalLM.from_pretrained(model_name,
                                                                device_map="auto",
                                                                trust_remote_code=True))
                self.refs.append(AutoModelForCausalLM.from_pretrained(model_name,
                                                                device_map="auto",
                                                                trust_remote_code=True))
            # load tokenizers
            for tokenizer_name in self.dpo_config.model_names:
                self.tokenizers.append(AutoTokenizer.from_pretrained(tokenizer_name,
                                                          device_map="auto",
                                                          quantization_config=bnb_config))
        logging.info("**LOADED AND INITIALIZED MODELS...**")
        subprocess.run(['nvidia-smi'])

    # full train
    def _train_individual(self,model_idx):
        subprocess.run(['nvidia-smi'])
        # load model
        policy = self.policies[model_idx]
        ref = self.refs[model_idx]
        tokenizer = self.tokenizers[model_idx]
        subprocess.run(['nvidia-smi'])
        # set optimizer
        self.optimizer = torch.optim.AdamW(policy.parameters(),
                                           lr=self.dpo_config.lr)
        # model in training mode
        policy.train()
        bs = self.dpo_config.batch_size
        # set up training
        for epoch in range(self.dpo_config.num_epochs):
            subprocess.run(['nvidia-smi'])
            logging.info("epoch: f'{epoch}' ")
            for batch_ in self.train_data:
                logging.info('new batch...')
                # tokenize; grab batch items; batch is like {transcript: [], note0: [], ...}, list is size of batch
                # also move data over the GPUs lest we overwhelm those CPUs
                subprocess.run(['nvidia-smi'])
                ids_queries = (tokenizer(batch_['transcript'],return_tensors='pt').input_ids)
                ids_note0s = (tokenizer(batch_['note0'],return_tensors='pt').input_ids)
                ids_note1s = (tokenizer(batch_['note1'],return_tensors='pt').input_ids)
                ids_note2s = (tokenizer(batch_['note2'],return_tensors='pt').input_ids)
                subprocess.run(['nvidia-smi'])
                # concatenate query and response tensors
                tensors_note0 = torch.cat((ids_queries,ids_note0s),dim=-1)
                tensors_note1 = torch.cat((ids_queries,ids_note1s),dim=-1)
                tensors_note2 = torch.cat((ids_queries,ids_note2s),dim=-1)
                del ids_queries, ids_note0s, ids_note1s, ids_note2s
                subprocess.run(['nvidia-smi'])
                logging.info("forward pass on policy model")
                # forward pass; attend all tokens
                logits_note0 = policy(**{"input_ids":tensors_note0,"attention_mask":torch.ones_like(tensors_note0)}).logits
                # Compute the total size in bytes
                subprocess.run(['nvidia-smi'])
                torch.cuda.empty_cache()
                logits_note0 = logits_note0.detach()
                logits_note0.to("cpu")
                subprocess.run(['nvidia-smi'])
                logits_note1 = policy(**{"input_ids":tensors_note1,"attention_mask":torch.ones_like(tensors_note1)}).logits
                logits_note1 = logits_note1.detach()
                logits_note1.to("cpu")
                subprocess.run(['nvidia-smi'])
                torch.cuda.empty_cache()
                logits_note2 = policy(**{"input_ids":tensors_note2,"attention_mask":torch.ones_like(tensors_note2)}).logits
                logits_note2 = logits_note2.detach()
                logits_note2.to("cpu")
                subprocess.run(['nvidia-smi'])
                torch.cuda.empty_cache()

                policy_logprobs = [process_logits(logits_note0),process_logits(logits_note1),process_logits(logits_note2)]
                del logits_note0, logits_note1, logits_note2
                subprocess.run(['nvidia-smi'])

                logging.info("finished forward pass on policy...")
                with torch.no_grad():
                    logging.info("forward pass on reference model")
                    ref_logits_note0 = ref(**{"input_ids":tensors_note0,"attention_mask":torch.ones_like(tensors_note0)}).logits
                    ref_logits_note0 = ref_logits_note0.detach()
                    ref_logits_note0.to("cpu")
                    logging.info("finished forward on note0")
                    ref_logits_note1 = ref(**{"input_ids":tensors_note1,"attention_mask":torch.ones_like(tensors_note1)}).logits
                    ref_logits_note1 = ref_logits_note1.detach()
                    ref_logits_note1.to("cpu")
                    logging.info("finished forward on note1")
                    ref_logits_note2 = ref(**{"input_ids":tensors_note2,"attention_mask":torch.ones_like(tensors_note2)}).logits
                    ref_logits_note2 = ref_logits_note2.detach()
                    ref_logits_note2.to("cpu")
                    subprocess.run(['nvidia-smi'])
                    logging.info("finished forward pass on reference...")
                del tensors_note0, tensors_note1, tensors_note2
                # as a note: logits shape (batch size, seq len, vocab size) –– so we want to keep just the last row
                # of the sequence dimension since it will include the logits for the full sequence. this way we have
                # (assuming batch size 1) a logit item of size [1,1,vocab_size]. can go ahead and aggregate the logprobs
                # of this item for computing the DPO loss
                ref_logprobs = [process_logits(ref_logits_note0[:, -1:, :]),
                                process_logits(ref_logits_note1[:, -1:, :]),
                                process_logits(ref_logits_note2[:, -1:, :])]
                del ref_logits_note0, ref_logits_note1, ref_logits_note2

                #if self.dpo_config.loss == "individual":
                subprocess.run(['nvidia-smi'])
                self.optimizer.zero_grad()
                logging.info("compute dpo loss")
                # grab pref ranking (todo adjust for batch size > 1)
                pref_ranking = format_pref_ranking([batch_['rank1'][0],batch_['rank2'][0],batch_['rank3'][0]])
                loss = individual_loss_dpo(policy_logprobs,ref_logprobs,pref_ranking)
                print('loss value: ',loss)
                # loss = loss.detach()
                # loss = loss.to(device="cuda")
                # loss.backward()
                # subprocess.run(['nvidia-smi'])
                # self.optimizer.step()
                # subprocess.run(['nvidia-smi'])
                logging.info("done with batch!")

        print('___________finished test train!___________')
        torch.cuda.empty_cache()
        # save the policy model
        models_folname = "/home/agunal/scratch/goldkind-clinical-ai/dev/ClinicalEval/saved_models/"
        model_path = models_folname + "policy_llama8b"
        torch.save(policy,model_path)


def format_pref_ranking(init_pr):
    res_pr = [-1,-1,-1]
    for idx,item in enumerate(init_pr):
        if '0' in item:
            res_pr[idx] = 1
        if '1' in item:
            res_pr[idx] = 2
        if '2' in item:
            res_pr[idx] = 3
    return res_pr


'''
logprobs are processed before this function (i.e. logits -> logprobs -> single scalar)
params:
{}_logprobs = list of tensors, tensor at index i corresponds to aggregate logprobs for the ith ranked note (so should be a single scalar value)
pref_ranking = dictionary mapping ranks to notes (so {1:note0, ...})
'''
def individual_loss_dpo(policy_logprobs,ref_logprobs,pref_ranking,beta=.5):
    print('policy logprobs: ',policy_logprobs)
    print('ref logprobs: ',ref_logprobs)
    loss = 1.0
    # order notes by rank
    pref_ranking = sorted(pref_ranking, key=lambda x: int(x), reverse=False)
    for k, rank in enumerate(pref_ranking):
        numerator = math.exp(beta * (policy_logprobs[rank] - ref_logprobs[rank]))
        denominator = 0.0
        for j, rank2 in enumerate(range(k,len(pref_ranking))):
            denominator += math.exp(beta * (policy_logprobs[rank2] - ref_logprobs[rank2]))
        loss *= (numerator / denominator)
    return loss

def process_logits(logits):
    print('logits before processing: ',logits, 'shape: ',logits.shape)
    log_probs = torch.log(F.log_softmax(logits, dim=2))
    agg_log_probs = log_probs.sum(dim=2)
    print('logits after processing: ',agg_log_probs, 'shape: ',logits.shape)
    return agg_log_probs



import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional
import torch.nn.functional as F
import tqdm
import sys
#from accelerate import Accelerator
import os

import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
import subprocess

from trainer_utils import *
from eval_utils import *

class DPOTrainer():
    def __init__(self,dpo_config,data_dict):
        self.dpo_config = dpo_config
        self.policies, self.refs, self.tokenizers = [], [], []
        self.policy_objs = []
       # self.accelerator = Accelerator()
        self.train_data = data_dict['train']
        self.eval_data = data_dict['eval']

    def load_policies(self,):
        for model_idx, model_name in enumerate(self.dpo_config.model_names):
            self.policy_objs.append(AutoModelForCausalLM.from_pretrained(model_name,
                                                                    device_map="auto",
                                                                    trust_remote_code=True))

    # load models (policies) and maintain original copies (ref)
    # todo: this code is a little outdated since I only use it to load a single policy/ref model
    # for individual training
    def load_models(self,):
        quick_status("loading reference and policy models...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
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



    # full train
    def _train_individual(self,model_idx):
        # load model
        policy = self.policies[model_idx]
        for param in policy.parameters():
            param.requires_grad_(True)
        ref = self.refs[model_idx]
        tokenizer = self.tokenizers[model_idx]
        # set optimizer
        optimizer = torch.optim.AdamW(policy.parameters(),
                                      lr=self.dpo_config.lr)
        # model in training mode
        policy.train()
        # set up training
        losses = []
        quick_status("begin training")
        for epoch in range(self.dpo_config.num_epochs):
            quick_status(f"beginning epoch: {epoch}")
            for batch_ in self.train_data:
                # tokenize; grab batch items; batch is like {transcript: [], note0: [], ...}, list is size of batch
                ids_queries = (tokenizer(batch_['transcript'],return_tensors='pt',max_length=self.dpo_config.tokenizer_max_len).input_ids)
                ids_note0s = (tokenizer(batch_['note0'],return_tensors='pt',max_length=self.dpo_config.tokenizer_max_len).input_ids)
                ids_note1s = (tokenizer(batch_['note1'],return_tensors='pt',max_length=self.dpo_config.tokenizer_max_len).input_ids)
                ids_note2s = (tokenizer(batch_['note2'],return_tensors='pt',max_length=self.dpo_config.tokenizer_max_len).input_ids)
                # concatenate query and response tensors
                tensors_note0 = torch.cat((ids_queries,ids_note0s),dim=-1)
                tensors_note1 = torch.cat((ids_queries,ids_note1s),dim=-1)
                tensors_note2 = torch.cat((ids_queries,ids_note2s),dim=-1)
                del ids_queries, ids_note0s, ids_note1s, ids_note2s
                # forward pass; attend all tokens
                logits_note0 = (policy(**{"input_ids":tensors_note0,"attention_mask":torch.ones_like(tensors_note0)}).logits)
                proc_logits_note0 = process_logits(logits_note0[:, -1:, torch.squeeze(tensors_note0)])
                del logits_note0
                logits_note1 = policy(**{"input_ids":tensors_note1,"attention_mask":torch.ones_like(tensors_note1)}).logits
                proc_logits_note1 = process_logits(logits_note1[:, -1:, torch.squeeze(tensors_note1)])
                del logits_note1
                logits_note2 = policy(**{"input_ids":tensors_note2,"attention_mask":torch.ones_like(tensors_note2)}).logits
                proc_logits_note2 = process_logits(logits_note2[:, -1:, torch.squeeze(tensors_note2)])

                del logits_note2
                policy_logprobs = [proc_logits_note0,proc_logits_note1,proc_logits_note2]

                with torch.no_grad():
                    ref_logits_note0 = ref(**{"input_ids":tensors_note0,"attention_mask":torch.ones_like(tensors_note0)}).logits
                    proc_ref_logits_note0 = process_logits(ref_logits_note0[:, -1:, torch.squeeze(tensors_note0)])
                    del ref_logits_note0
                    ref_logits_note1 = ref(**{"input_ids":tensors_note1,"attention_mask":torch.ones_like(tensors_note1)}).logits
                    proc_ref_logits_note1 = process_logits(ref_logits_note1[:, -1:, torch.squeeze(tensors_note1)])
                    del ref_logits_note1
                    ref_logits_note2 = ref(**{"input_ids":tensors_note2,"attention_mask":torch.ones_like(tensors_note2)}).logits
                    proc_ref_logits_note2 = process_logits(ref_logits_note2[:, -1:, torch.squeeze(tensors_note2)])
                    del ref_logits_note2, tensors_note0, tensors_note1, tensors_note2

                # as a note: logits shape (batch size, seq len, vocab size) –– so we want to keep just the last row
                # of the sequence dimension since it will include the logits for the full sequence. this way we have
                # (assuming batch size 1) a logit item of size [1,1,vocab_size]. can go ahead and aggregate the logprobs
                # of this item for computing the DPO loss
                ref_logprobs = [proc_ref_logits_note0,proc_ref_logits_note1,proc_ref_logits_note2]
                # grab pref ranking (TODO adjust for batch size > 1)
                pref_ranking = format_pref_ranking([batch_['rank1'][0],batch_['rank2'][0],batch_['rank3'][0]])
                loss = individual_loss_dpo(policy_logprobs,ref_logprobs,pref_ranking)

                quick_status(f'Loss: {loss}')
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                quick_status(f'Epoch [{epoch+1}/{self.dpo_config.num_epochs}], Loss: {loss.item()}')
                losses.append(loss.item())

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss During Training')
        plt.legend()
        plt.savefig("traininglosses")

        quick_status("training complete! deleting models from GPU and writing to hub...")
        torch.cuda.empty_cache()
        # save the policy model on hf to save disk space
        try:
            _name = "aegunal/in_" + self.dpo_config.map_model_names[model_idx]
            policy.push_to_hub(_name)
            # delete model to free up GPU space
            del policy, ref, tokenizer
            return _name
        except Exception as e:
            print("tried uploading to hub, here is the error: ", e)
            # delete model to free up GPU space
            del policy, ref, tokenizer
            return None

    '''
    - load model from last checkpoint, if it exists
    - fine-tune it on the next batch
    - save model checkpoint
    - free model from gpu space
    '''
    def _train_ensemble(self,):
        # load policy models
        quick_status("loading models...")
        self.load_policies()
        losses = [[],[],[]]
        train_step = 0
        for epoch in range(self.dpo_config.num_epochs):
            quick_status('new epoch...')
            print('training data: ',len(self.train_data))
            for batch_ in self.train_data:
                quick_status('new batch...')
                cur_losses = []
                # iter through the models to compute loss
                for model_idx, _ in enumerate(self.dpo_config.model_names):
                    torch.cuda.empty_cache()
                    quick_status(f"training {_} on step {train_step}")
                    # load nec models
                    model_name = self.dpo_config.model_names[model_idx]
                    # ref and tokenizer can be initialized from pre-trained
                    quick_status(f"loading reference model {model_name} and tokenizer...")
                    ref = AutoModelForCausalLM.from_pretrained(model_name,
                                                               device_map="auto",
                                                               trust_remote_code=True)
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                            device_map="auto",
                                                            quantization_config=bnb_config)
                    # get inputs
                    quick_status("processing inputs...")
                    ids_queries = (tokenizer(batch_['transcript'],return_tensors='pt',max_length=self.dpo_config.tokenizer_max_len).input_ids)
                    ids_note0s = (tokenizer(batch_['note0'],return_tensors='pt',max_length=self.dpo_config.tokenizer_max_len).input_ids)
                    ids_note1s = (tokenizer(batch_['note1'],return_tensors='pt',max_length=self.dpo_config.tokenizer_max_len).input_ids)
                    ids_note2s = (tokenizer(batch_['note2'],return_tensors='pt',max_length=self.dpo_config.tokenizer_max_len).input_ids)
                    # concatenate query and response tensors
                    tensors_note0 = torch.cat((ids_queries,ids_note0s),dim=-1)
                    tensors_note1 = torch.cat((ids_queries,ids_note1s),dim=-1)
                    tensors_note2 = torch.cat((ids_queries,ids_note2s),dim=-1)
                    del ids_queries, ids_note0s, ids_note1s, ids_note2s
                    # forward pass; attend all tokens
                    quick_status("forward pass with policy model...")
                    logits_note0 = self.policy_objs[model_idx](**{"input_ids":tensors_note0,"attention_mask":torch.ones_like(tensors_note0)}).logits
                    proc_logits_note0 = process_logits(logits_note0[:, -1:, torch.squeeze(tensors_note0)])
                    del logits_note0
                    logits_note1 = self.policy_objs[model_idx](**{"input_ids":tensors_note1,"attention_mask":torch.ones_like(tensors_note1)}).logits
                    proc_logits_note1 = process_logits(logits_note1[:, -1:, torch.squeeze(tensors_note1)])
                    del logits_note1
                    logits_note2 = self.policy_objs[model_idx](**{"input_ids":tensors_note2,"attention_mask":torch.ones_like(tensors_note2)}).logits
                    proc_logits_note2 = process_logits(logits_note2[:, -1:, torch.squeeze(tensors_note2)])
                    del logits_note2
                    policy_logprobs = [proc_logits_note0,proc_logits_note1,proc_logits_note2]
                    # forward pass on ref model
                    with torch.no_grad():
                        quick_status("forward pass with reference model....")
                        ref_logits_note0 = ref(**{"input_ids":tensors_note0,"attention_mask":torch.ones_like(tensors_note0)}).logits
                        proc_ref_logits_note0 = process_logits(ref_logits_note0[:, -1:, torch.squeeze(tensors_note0)])
                        del ref_logits_note0
                        ref_logits_note1 = ref(**{"input_ids":tensors_note1,"attention_mask":torch.ones_like(tensors_note1)}).logits
                        proc_ref_logits_note1 = process_logits(ref_logits_note1[:, -1:, torch.squeeze(tensors_note1)])
                        del ref_logits_note1
                        ref_logits_note2 = ref(**{"input_ids":tensors_note2,"attention_mask":torch.ones_like(tensors_note2)}).logits
                        proc_ref_logits_note2 = process_logits(ref_logits_note2[:, -1:, torch.squeeze(tensors_note2)])
                        del ref_logits_note2, tensors_note0, tensors_note1, tensors_note2
                    ref_logprobs = [proc_ref_logits_note0,proc_ref_logits_note1,proc_ref_logits_note2]
                    # grab pref ranking (again TODO adjust for batch size > 1)
                    pref_ranking = format_pref_ranking([batch_['rank1'][0],batch_['rank2'][0],batch_['rank3'][0]])
                    # compute loss
                    loss = individual_loss_dpo(policy_logprobs,ref_logprobs,pref_ranking)
                    cur_losses.append(loss)
                    quick_status(f'{self.dpo_config.map_model_names[model_idx]} loss: {loss}')
                    # at this point we can discard ref and tokenizer for current model
                    del ref, tokenizer
                # compute the aggregate loss
                agg_loss = (torch.stack(cur_losses)).mean()
                for model_idx, _ in enumerate(self.dpo_config.model_names):
                    # load optimizer
                    checkpts_folname = "/home/agunal/scratch/goldkind-clinical-ai/dev/ClinicalEval/pl/optim_checkpts/"
                    checkpts_optim_fname = checkpts_folname + self.dpo_config.map_model_names[model_idx] + '_optim'
                    # create optim file
                    if os.path.exists(checkpts_optim_fname) == False:
                        with open(checkpts_optim_fname,'w+') as outf:
                            pass
                    optimizer = torch.optim.AdamW(self.policy_objs[model_idx].parameters(),
                                                  lr=self.dpo_config.lr)
                    if train_step != 0:
                        optim_checkpt = torch.load(checkpts_optim_fname)
                        optimizer.load_state_dict(optim_checkpt['optimizer_state_dict'])
                    # zero the gradients
                    optimizer.zero_grad()
                # combined backprop
                agg_loss.backward()
                # load optimizers again and take step for each model
                for model_idx, _ in enumerate(self.dpo_config.model_names):
                    # load optimizer
                    checkpts_folname = "/home/agunal/scratch/goldkind-clinical-ai/dev/ClinicalEval/pl/optim_checkpts/"
                    checkpts_optim_fname = checkpts_folname + self.dpo_config.map_model_names[model_idx] + '_optim'
                    # create optim file
                    if os.path.exists(checkpts_optim_fname) == False:
                        with open(checkpts_optim_fname,'w+') as outf:
                            pass
                    optimizer = torch.optim.AdamW(self.policy_objs[model_idx].parameters(),
                                                  lr=self.dpo_config.lr)
                    if train_step != 0:
                        optim_checkpt = torch.load(checkpts_optim_fname)
                        optimizer.load_state_dict(optim_checkpt['optimizer_state_dict'])
                    optimizer.step()
                    quick_status("saving optimizer checkpoint....")
                    torch.save({'optimizer_state_dict':optimizer.state_dict(), 'loss':loss}, checkpts_optim_fname)
                    # delete everything to free up GPU space
                    quick_status("end of a step; freeing up GPU space by deleting non-policy models")
                    losses[model_idx].append(loss)
                    del optimizer
            break


    def _eval_ensemble(self):
        quick_status("********EVALUATION TIME*********")
        pldata = PLEvalData()
        train_dataloader, eval_dataloader = load_data(pldata, batch_size=self.dpo_config.batch_size,split=.85,seed=123)
        data_dict = {'train':train_dataloader, 'eval':eval_dataloader}
        # load eval models
        quick_status("loading scoring models...")
        evaluator0 = ClinicalNotesEval(model=self.policy_objs[0],tokenizer=AutoTokenizer.from_pretrained(self.dpo_config.model_names[0]),train_dl=train_dataloader,eval_dl=eval_dataloader)
        evaluator1 = ClinicalNotesEval(model=self.policy_objs[1],tokenizer=AutoTokenizer.from_pretrained(self.dpo_config.model_names[0]),train_dl=train_dataloader,eval_dl=eval_dataloader)
        evaluator2 = ClinicalNotesEval(model=self.policy_objs[2],tokenizer=AutoTokenizer.from_pretrained(self.dpo_config.model_names[0]),train_dl=train_dataloader,eval_dl=eval_dataloader)
        # train eval models on eval task
        quick_status("training eval models...")
        evaluator0._train()
        evaluator1._train()
        evaluator2._train()
        # aggregate preds
        from sklearn.metrics import accuracy_score
        # evaluate
        golds = []
        preds = []        
        for batch_ in eval_dataloader:
            golds, preds0 = evaluator0._get_pred()
            _, preds1 = evaluator1._get_pred()
            _, preds2 = evaluator2._get_pred()
            pred0 = round(((float(preds0[0]) + float(preds1[0]) + float(preds2[0]))) / 3.0)
            pred1 = round(((float(preds0[1]) + float(preds1[1]) + float(preds2[1]))) / 3.0)
            pred2 = round(((float(preds0[2]) + float(preds1[2]) + float(preds2[2]))) / 3.0)
            golds.extend(golds)
            preds.extend([pred0,pred1,pred2])

        return accuracy_score(golds,preds)


def format_pref_ranking(init_pr):
    res_pr = [-1,-1,-1]
    for idx,item in enumerate(init_pr):
        if '0' in item:
            res_pr[idx] = 0
        if '1' in item:
            res_pr[idx] = 1
        if '2' in item:
            res_pr[idx] = 2
    return res_pr

# logits --> logprobs --> aggregate
def process_logits(logits):
    log_probs = F.log_softmax(logits, dim=2)
    agg_log_probs = torch.flatten(log_probs.mean(dim=2))
    return agg_log_probs

# logging
def quick_status(msg):
    logging.info(f"{msg}")
    subprocess.run(['nvidia-smi'])





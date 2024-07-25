
import torch
import math
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from accelerate import Accelerator

class DPOTrainer():
    def __init__(self,dpo_config,data_dict):
        self.dpo_config = dpo_config
        self.policies, self.refs, self.tokenizers = [], [], []
        self.train_data = data_dict['train']
        self.eval_data = data_dict['eval']
        self.accelerator = Accelerator()
        self.optimizer = AdamW()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load models (policies) and maintain original copies (ref)
    def load_models(self,):
        # don't re-load if you don't need to!!
        if len(self.policies) == 0 and len(self.refs) == 0 and len(self.tokenizers) == 0:
            for model_name in self.dpo_config.model_names:
                pol_model = AutoModelForCausalLM.from_pretrained(model_name)
                self.policies.append(pol_model)
                ref_model = AutoModelForCausalLM.from_pretrained(model_name)
                self.refs.append(ref_model)
            # load tokenizers
            for tokenizer_name in self.dpo_config.model_names:
                tokenizer = AutoModelForCausalLM.from_pretrained(tokenizer_name)
                self.tokenizers.append(tokenizer_name)
        # move stuff over to accelerator TODO

    # training step
    def step(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        preference_mask: Optional[torch.BoolTensor] = None,
    ):
        self.model.train()
        bs = self.dpo_config.batch_size

        for i in tqdm.tqdm(range(0, bs), desc="Training in batches", leave=False):
            queries_ = queries[i : i + bs]
            responses_w_ = responses_w[i : i + bs]
            responses_l_ = responses_l[i : i + bs]
            preference_mask_ = preference_mask[i : i + bs] if preference_mask is not None else None

            loss, stats = self._step(
                queries=queries_,
                responses_w=responses_w_,
                responses_l=responses_l_,
                return_stats=True,
                preference_mask=preference_mask_,
            )
            
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.current_step += 1
        return stats


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





''' define the various loss functions '''

import torch

'''
logprobs are processed before this function (i.e. logits -> logprobs -> single scalar)
params:
{}_logprobs = list of tensors, tensor at index i corresponds to aggregate logprobs for the ith ranked note (so should be a single scalar value)
pref_ranking = dictionary mapping ranks to notes (so {1:note0, ...})
'''
def individual_loss_dpo(policy_logprobs,ref_logprobs,pref_ranking,beta=.001):
    loss = torch.tensor(1.0, dtype=torch.float32)#, requires_grad=True)
    # order notes by rank
    pref_ranking = sorted(pref_ranking, key=lambda x: int(x), reverse=False)
    for k, rank in enumerate(pref_ranking):
        numerator = torch.exp(beta * (policy_logprobs[rank][0] - ref_logprobs[rank][0]))
        print('num: pol logprobs idx item: ',policy_logprobs[rank][0], 'ref logprobs idx item: ', ref_logprobs[rank][0])
        denominator = torch.tensor(0.0, dtype=torch.float32)
        for j in range(k,len(pref_ranking)):
            rank2 = pref_ranking[j]
            denominator += torch.exp(beta * (policy_logprobs[rank2][0] - ref_logprobs[rank2][0]))
            print('denom: pol logprobs idx item: ',policy_logprobs[rank2][0], 'ref logprobs idx item: ', ref_logprobs[rank2][0])
        print('numerator: ',numerator,' || denoninator: ',denominator)
        loss *= (numerator / denominator)
    print('loss before log: ',loss)
    return torch.neg(torch.log(loss))

























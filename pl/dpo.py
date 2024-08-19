

# my modules
from data_utils import *
from dpo_config import *
from dpo_trainer import *

# libraries
import logging
import sys

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

# hf auth
from auth_tokens import HF_TOKEN
from huggingface_hub import login
login(token=HF_TOKEN)

### set config
# logistics
config = DPOConfig()
config.model_names=[# "mistralai/Mistral-7B-v0.3",
                    "meta-llama/Meta-Llama-3-8B",
                 #    "google/gemma-7b",
                 #    "microsoft/Phi-3-small-128k-instruct"
                    ]
config.tokenizer_names=[#"mistralai/Mistral-7B-v0.3",
                        "meta-llama/Meta-Llama-3-8B",
                     #   "google/gemma-7b",
                     #   "microsoft/Phi-3-small-128k-instruct"
                        ]
config.output_dir="/home/agunal/scratch/goldkind-clinical-ai/tmpdir/job_outputs"
# training details
config.train_method="individual"
config.loss_method="individual_loss"
config.num_epochs=1
config.batch_size=1
config.lr=.01

def main():
  # initialize data
  subprocess.run(['nvidia-smi'])
  logging.info("loading and processing dataset...")
  pldata = PLData()
  subprocess.run(['nvidia-smi'])
  train_dataloader, eval_dataloader = load_data(pldata,batch_size=config.batch_size,split=.85)
  subprocess.run(['nvidia-smi'])
  data_dict = {'train':train_dataloader,'eval':eval_dataloader} # one instance will have dimension batch_size per item (e.g. transcript, rank1, etc)
  # load models and tokenizers
  models = []
  tokenizers = []
  # initialize trainer obj
  dpo_trainer_ = DPOTrainer(config,data_dict)
  subprocess.run(['nvidia-smi'])
  logging.info("loading evaluator models...")
  dpo_trainer_.load_models()
  subprocess.run(['nvidia-smi'])

  if config.train_method == "individual":
      logging.info("training models individually...")
      for idx,model_name in enumerate(config.model_names):
          dpo_trainer_._train_individual(idx)
  logging.info("training complete!")

if __name__ == "__main__":
    main()
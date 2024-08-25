

# my modules
from data_utils import *
from dpo_config import *
from dpo_trainer import *
from config_utils import *
from dpo_eval import *

# libraries
import logging
import sys

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

# hf auth
from auth_tokens import HF_TOKEN_RW
from huggingface_hub import login
login(token=HF_TOKEN_RW)

def main():
  # initializing dpo config
  config = DPOConfig()
  config = populate_config(config)
  # initialize data
  logging.info("loading and processing dataset...")
  pldata = PLData()
  train_dataloader, eval_dataloader = load_data(pldata, batch_size=config.batch_size,split=.85)
  data_dict = {'train':train_dataloader, 'eval':eval_dataloader} # one instance will have dimension batch_size per item (e.g. transcript, rank1, etc)
  # load models and tokenizers
  models = []
  tokenizers = []
  # initialize trainer obj
  dpo_trainer_ = DPOTrainer(config,data_dict)
  logging.info("loading evaluator models...")
  dpo_trainer_.load_models()

  if config.train_method == "individual":
      logging.info("training models individually...")
      for idx,model_name in enumerate(config.model_names):
          dpo_trainer_._train_individual(idx)
  logging.info("training complete!")
  # evaluation
  dpo_evaluator = ClinicalNotesEval()
      




if __name__ == "__main__":
    main()







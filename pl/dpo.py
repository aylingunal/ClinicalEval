

# my modules
from data_utils import *
from dpo_config import *
from dpo_trainer import *

# libraries 
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(levelname)s - %(message)s')

# hf auth
from auth_tokens import HF_TOKEN
from huggingface_hub import login
login(token=HF_TOKEN)

### set config
# logistics
config = DPOConfig()
config.model_names=["mistralai/Mistral-7B-v0.3",
                    "meta-llama/Meta-Llama-3-8B",
                    "google/gemma-7b",
                    "microsoft/Phi-3-small-128k-instruct"
                    ]
config.tokenizer_names=["mistralai/Mistral-7B-v0.3",
                        "meta-llama/Meta-Llama-3-8B",
                        "google/gemma-7b",
                        "microsoft/Phi-3-small-128k-instruct"
                        ]
config.output_dir="/home/agunal/scratch/goldkind-clinical-ai/tmpdir/job_outputs"
# training details
config.train_method="individual"
config.loss_method="individual_loss"
config.num_epochs=5
config.batch_size=4
config.lr=.01


def main():
    # initialize data
    logging.info("loading and processing dataset...")
    pldata = PLData()
    train_dataloader, eval_dataloader = load_data(pldata,batch_size=config.batch_size,split=.85)
    data_dict = {'train':train_dataloader,'eval':eval_dataloader} # one instance will have dimension batch_size per item (e.g. transcript, rank1, etc)
    # load models and tokenizers
    models = []
    tokenizers = []
    # initialize trainer obj
    dpo_trainer = DPOTrainer(config,data_dict)
    logging.info("loading evaluator models...")
    dpo_trainer.load_models()

    if config.train_method == "individual":
        logging.info("training models individually...")
        for idx,model_name in enumerate(config.model_names):
            dpo_trainer._train_individual(idx)


if __name__ == "__main__":
    main()
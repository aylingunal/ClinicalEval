# config parameters
model_names=[ #"mistralai/Mistral-7B-v0.3",
                    "meta-llama/Meta-Llama-3-8B",
                  #   "google/gemma-7b",
                 #    "microsoft/Phi-3-small-128k-instruct"
                    ]
tokenizer_names=[#"mistralai/Mistral-7B-v0.3",
                        "meta-llama/Meta-Llama-3-8B",
                    #    "google/gemma-7b",
                     #   "microsoft/Phi-3-small-128k-instruct"
                        ]
tokenizer_max_len = 1000
output_dir="/home/agunal/scratch/goldkind-clinical-ai/tmpdir/job_outputs"
# training details
train_method="individual"
loss_method="individual_loss"
num_epochs=5
batch_size=1
lr=.01

# populate
def populate_config(_config):
    _config.model_names = model_names
    _config.tokenizer_names = tokenizer_names
    _config.tokenizer_max_len = tokenizer_max_len
    _config.output_dir = output_dir
    _config.train_method = train_method
    _config.loss_method = loss_method
    _config.num_epochs = num_epochs
    _config.batch_size = batch_size
    _config.lr = lr
    return _config




from eval_utils import *
from config_utils import *
from dpo_config import *

def main():
    config = DPOConfig()
    config = populate_config(config)
    pldata = PLEvalData()
    train_dataloader, eval_dataloader = load_data(pldata, batch_size=config.batch_size,split=.85,seed=123)
    data_dict = {'train':train_dataloader, 'eval':eval_dataloader} # one instance will have dimension batch_size per item (e.g. transcript, rank1, etc)


    return


if __name__=='__main__':
    main()



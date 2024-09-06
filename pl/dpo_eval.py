''' run this script strictly on the baselines; 
for getting PL-tuned results, need to run dpo.py '''


from eval_utils import *
from config_utils import *
from dpo_config import *

def main():
    # dpo config stuff prob unnec for this context 
    config = DPOConfig()
    config = populate_config(config,TRAIN_HF_MODEL_NAMES,MAP_MODEL_NAMES)
    pldata = PLEvalData()
    train_ds, eval_ds = load_dataset(pldata, batch_size=config.batch_size,split=.85,seed=123)

    tmp = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-4B",device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B")

    obj = ClinicalNotesEval(tmp,tokenizer,train_ds,eval_ds)
    acc = obj._train_and_eval()
    print('backup acc: ',acc)

if __name__=='__main__':
    main()



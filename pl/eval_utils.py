import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_metric

from data_utils import *
from model_urls import *

''' for fitting a classification layer on top of fine-tuned policy model '''
class ClinicalNoteScorer(nn.Module):
    def __init__(self, base_model, num_labels):
        super(ClinicalNoteScorer, self).__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.config = base_model.config
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, lbl):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        # if evaluating, None 
        if lbl is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), lbl.view(-1))

        return SequenceClassifierOutput(loss, logits)

class ClinicalNotesEval():
    def __init__(self,model,train_dl,eval_dl):
        self.tokenizer = AutoTokenizer.from_pretrained("aegunal/ClinicalEval_llama8b")
        self.model = AutoModelForCausalLM.from_pretrained("aegunal/ClinicalEval_llama8b")
        self.scoring_model = ClinicalNoteScorer(self.model,num_labels=4)
        self.train_dl = train_dl
        self.eval_dl = eval_dl

    def _train(self):
        training_args = TrainingArguments(output_dir="test_trainer", 
                                        #  evaluation_strategy="epoch",
                                        num_train_epochs=5,
                                        logging_steps=50,)
        metric = load_metric('accuracy')
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels) 
    
        trainer = Trainer(model=self.scoring_model,
                          args=training_args,
                          train_dataset=self.train_dl,
                          compute_metrics=compute_metrics)
        trainer.train()

    def _eval(self):
        from sklearn.metrics import accuracy_score
        # evaluate
        golds = []
        preds = []
        for batch_ in self.eval_dl:
            # grab text items
            ids_queries = self.tokenizer(batch_['transcript'],return_tensors='pt',max_length=self.dpo_config.tokenizer_max_len).input_ids.unsqueeze(0)
            ids_note0s = self.tokenizer(batch_['note0'],return_tensors='pt',max_length=self.dpo_config.tokenizer_max_len).input_ids.unsqueeze(0)
            ids_note1s = self.tokenizer(batch_['note1'],return_tensors='pt',max_length=self.dpo_config.tokenizer_max_len).input_ids.unsqueeze(0)
            ids_note2s = self.tokenizer(batch_['note2'],return_tensors='pt',max_length=self.dpo_config.tokenizer_max_len).input_ids.unsqueeze(0)
            tensors_note0 = torch.cat((ids_queries,ids_note0s),dim=-1)
            tensors_note1 = torch.cat((ids_queries,ids_note1s),dim=-1)
            tensors_note2 = torch.cat((ids_queries,ids_note2s),dim=-1)
            # grab scores
            gold0 = batch_['note0']
            gold1 = batch_['note1']
            gold2 = batch_['note2']
            # model preds
            pred0 = torch.argmax(self.scoring_model(input_ids=tensors_note0,
                                                    attention_mask=torch.ones_like(tensors_note0),
                                                    lbl=None)['logits'].squeeze(0)).item()
            pred1 = torch.argmax(self.scoring_model(input_ids=tensors_note1,
                                                    attention_mask=torch.ones_like(tensors_note0),
                                                    lbl=None)['logits'].squeeze(0)).item()
            pred2 = torch.argmax(self.scoring_model(input_ids=tensors_note2,
                                                    attention_mask=torch.ones_like(tensors_note0),
                                                    lbl=None)['logits'].squeeze(0)).item()
            golds.extend([gold0,gold1,gold2])
            preds.extend([pred0,pred1,pred2])

        return accuracy_score(golds,preds)

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
        self.config = self.base_model.config
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.config.vocab_size, num_labels)

    def forward(self, input_ids, attention_mask, labels):
        # note --> unsqueeze(,0) the args if working with batch size 1 (i.e. when not wrapping w/ dataloader)
        outputs = self.base_model(input_ids=input_ids,
                                  attention_mask=attention_mask)
        mean_pool = outputs.logits.mean(dim=1)
        # dropout
        dropped_out = self.dropout(mean_pool)
        # grab logits per category
        logits = self.classifier(dropped_out)
        loss = None
        # if evaluating, don't have to compute loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits,labels)
        del outputs,mean_pool,dropped_out
        return loss, SequenceClassifierOutput(logits)

class ClinicalNotesEval():
    def __init__(self,model,tokenizer,train_dl,eval_dl):
        self.tokenizer = tokenizer
        self.scoring_model = ClinicalNoteScorer(model,num_labels=4)
        self.train = train_dl
        self.eval = eval_dl
    
    def _format_data(self):
        class TrainData():
            def __init__(self,encodings,labels):
                self.encodings = encodings
                self.labels = labels
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
            def __len__(self):
                return len(self.labels)

        #### train
        note0_texts = [x['note0'] for x in self.train]
        note1_texts = [x['note1'] for x in self.train]
        note2_texts = [x['note2'] for x in self.train]
        note0_scores = [x['score0'] for x in self.train]
        note1_scores = [x['score1'] for x in self.train]
        note2_scores = [x['score2'] for x in self.train]

        note_texts = note0_texts + note1_texts + note2_texts
        note_encs = self.tokenizer(note_texts,max_length=750,truncation=True,padding=True,return_tensors="pt")
        note_scores = note0_scores + note1_scores + note2_scores
    
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        import math
        note_scores = [math.floor(x) for x in note_scores] # doing this so we can include some 2s in the mix
        le.fit(note_scores)
        note_scores_enc = le.transform(note_scores)
        train_dataset=TrainData(encodings=note_encs,labels=note_scores_enc)
        #### test
        note0_texts_test = [x['note0'] for x in self.eval]
        note1_texts_test = [x['note1'] for x in self.eval]
        note2_texts_test = [x['note2'] for x in self.eval]
        note0_scores_test = [x['score0'] for x in self.eval]
        note1_scores_test = [x['score1'] for x in self.eval]
        note2_scores_test = [x['score2'] for x in self.eval]

        note_texts_test = note0_texts_test + note1_texts_test + note2_texts_test
        note_encs_test = self.tokenizer(note_texts_test,max_length=750,truncation=True,padding=True,return_tensors="pt")
        note_scores_test = note0_scores_test + note1_scores_test + note2_scores_test
    
        note_scores_test = [math.floor(x) for x in note_scores_test] # doing this so we can include some 2s in the mix
        note_scores_enc_test = le.transform(note_scores_test)
        test_dataset=TrainData(encodings=note_encs_test,labels=note_scores_enc_test)

        batch_size = 16
        return DataLoader(train_dataset,batch_size=batch_size), DataLoader(test_dataset,batch_size=batch_size)

    def _train_and_eval(self):
        train_dl, test_dl = self._format_data()
        num_epochs = 1
        optimizer = torch.optim.AdamW(self.scoring_model.parameters(), lr=.01)
        print('begin training classifier')
        for epoch in range(num_epochs):
            for batch in train_dl:
                loss, _ = self.scoring_model.forward(input_ids=batch['input_ids'],
                                                  attention_mask=batch['attention_mask'],
                                                  labels=batch['labels'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print('begin evaluation')
        from sklearn.metrics import accuracy_score
        golds, preds = [], []
        for batch in test_dl:
            _, pred = self.scoring_model.forward(input_ids=batch['input_ids'],
                                                  attention_mask=batch['attention_mask'],
                                                  labels=None)
            golds.append(batch['labels'])
            preds.append(pred)
        print('evaluation complete! accuracy: ',accuracy_score(golds,preds))
        return accuracy_score(golds,preds)


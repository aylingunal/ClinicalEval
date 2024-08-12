import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

import logging
logger = logging.basicConfig(filename='annots_emb_learning', level=logging.INFO)

''' data loading + formatting '''
import json
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import random
from copy import deepcopy

ANNOT_SCORES_FNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/data/format_annots_scores_annotids.json"
ANNOTS_RANKS_FNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/data/format_annots_ranks_annotids.json"
NOTES_FOLNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/data/synth_notes/"

''' NN set-up '''
# base nn definition
class EmbedNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EmbedNN, self).__init__()
        self.embedding_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        embedding = self.embedding_layer(x)
        out = self.relu(embedding)
        out = self.fc2(out)
        return out, embedding

''' embedding model '''
class AnnotEmbedModel():
    def __init__(self,
                 hidden_dim=128) -> None:
        # misc inits
        self.hidden_dim = hidden_dim

    def init_sent_bert(self,
                       sent_emb_dim=16):
        logging.info("initializing sbert...")

        class CompressedSentBERT(nn.Module):
            def __init__(self):
                super(CompressedSentBERT, self).__init__()
                # load model and initialize the output layer
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.dense = nn.Linear(self.model.get_sentence_embedding_dimension(),sent_emb_dim)
                ### PCA code straight from sbert repo ###
                # train a PCA model; we'll use it to populate the weights in dense layer
                logging.info("SBERT: training PCA model")
                train_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")
                nli_sentences = train_dataset["sentence1"] + train_dataset["sentence2"]
                random.shuffle(nli_sentences)
                pca_train_sentences = nli_sentences[0:25]
                train_embeddings = self.model.encode(pca_train_sentences, convert_to_numpy=True)
                pca = PCA(n_components=sent_emb_dim)
                pca.fit(train_embeddings)
                pca_comp = np.asarray(pca.components_)
                # add the weights to the output layer
                self.dense.weight = torch.nn.Parameter(torch.tensor(pca_comp))
            def forward(self,input_text):
                original_emb = torch.Tensor(self.model.encode([input_text])[0])
                return torch.Tensor(self.dense(original_emb))

        self.sbert_model = CompressedSentBERT()

    # initialize main model that will construct annotator embeddings
    def init_embedding_model(self):
        print('type x train: ',type(self.x_train[0]))
        input_dim = int(self.x_train[0].size())
        output_dim = int(self.y_train[0].size())
        print('dims: ',input_dim,output_dim)
        self.model = EmbedNN(input_size=input_dim,
                             hidden_size=self.hidden_dim,
                             output_size=output_dim)

    def load_data(self):
        logging.info("loading scores, ranks, and notes...")
        # read in annotations
        with open(ANNOT_SCORES_FNAME,'r') as inf:
            scores = json.load(inf)
        with open(ANNOTS_RANKS_FNAME,'r') as inf:
            ranks = json.load(inf)
        # grab ``legal" notes
        legal_transcripts = []
        for ann in scores.keys():
            legal_transcripts.extend([x for x in scores[ann].keys()])
        legal_transcripts = list(set(legal_transcripts))
        # manually discarding some incomplete ones
        legal_transcripts.remove('anxiety-transcript436')
        legal_transcripts.remove('anxiety-transcript58')
        # read in notes
        notes = {}
        for transcript_folname in legal_transcripts:
            # add in order of notes
            notes_arrs = []
            with open(NOTES_FOLNAME + transcript_folname + '/note0.txt','r') as inf:
                note0_text = inf.read()
            with open(NOTES_FOLNAME + transcript_folname + '/note1.txt','r') as inf:
                note1_text = inf.read()
            with open(NOTES_FOLNAME + transcript_folname + '/note2.txt','r') as inf:
                note2_text = inf.read()

            # sbert sorts sentences based on length so can't batch encode
            notes_arrs.append(self.sbert_model(note0_text))
            notes_arrs.append(self.sbert_model(note1_text))
            notes_arrs.append(self.sbert_model(note2_text))

            full_arr = torch.cat(notes_arrs)
            notes[transcript_folname] = full_arr
        return scores, ranks, notes

    def process_data(self):
        logging.info("processing data into inputs (annotid,catid,notes_embs) and outputs (ranks,scores)...")
        scores, ranks, notes = self.load_data()
        # format for training
        x_train = []
        y_train = []
        # label encoders for nominals
        self.le_annot = LabelEncoder()
        self.le_annot.fit(list(scores.keys()))
        self.le_cats = LabelEncoder()
        self.le_cats.fit(['stressor','symptom','accuracy','readability'])
        for annot_id in scores.keys():
            for transcript_id in scores[annot_id].keys():
                if 'anxiety-transcript436' not in transcript_id and 'anxiety-transcript58' not in transcript_id:
                    for cat in scores[annot_id][transcript_id].keys():
                        # inputs
                        annot_id_le = self.le_annot.transform([annot_id])[0]
                        cat_id_le = self.le_cats.transform([cat])[0]

                        x_inst = np.concatenate([[annot_id_le],[cat_id_le],notes[transcript_id].detach().numpy()])
                        x_inst = torch.Tensor(x_inst)
                        # outputs
                        ranks_dict_ = deepcopy(ranks[annot_id][transcript_id][cat][0])
                        ranks_dict = swap_keys_vals(ranks_dict_)
                        ranks_emb = torch.Tensor(np.array([int(ranks_dict['note0']),int(ranks_dict['note1']),int(ranks_dict['note2'])]))
                        scores_dict = scores[annot_id][transcript_id][cat][0]
                        scores_emb = torch.Tensor(np.array([int(scores_dict['note0']),int(scores_dict['note1']),int(scores_dict['note2'])]))
                        
                        y_inst = torch.cat([ranks_emb,scores_emb])
                        x_train.append(x_inst)
                        y_train.append(y_inst)

        self.x_train = x_train
        self.y_train = y_train

    def train(self):
        logging.info("training the model...")
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        # training loop
        num_epochs = 50
        for epoch in range(num_epochs):
            # forward pass
            outputs, _ = self.model(self.x_train)
            loss = criterion(outputs,self.y_train)
            # backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # this assumes the embedding model has been fully trained
    def get_embs(self):
        logging.info("retrieving annotator embeddings...")
        annotator_embeddings = {}
        scores, ranks, notes = self.load_data()
        for annotator_id in scores.keys():
            cur_inputs = []
            for transcript_id in scores[annotator_id].keys():
                for cat in scores[annotator_id][transcript_id].keys():
                    annotator_id_ = self.le_annot.transform([annotator_id])[0]
                    cat_id_ = self.le_cats.transform([cat])[0]
                    input = torch.Tensor([annotator_id_,cat_id_,notes[transcript_id]])
                    cur_inputs.append(input)
            # grab all of one annotator's embeddings
            cur_embs = []
            for input in cur_inputs:
                with torch.no_grad():
                    _, embedding = self.model(input)
                    cur_embs.append(embedding)
            # aggregate and normalize
            aggregate_emb = np.sum(cur_embs,axis=0)
            norm_emb = np.divide(aggregate_emb,len(cur_embs))
            annotator_embeddings[annotator_id] = norm_emb
        return annotator_embeddings

    def run(self):
        self.init_sent_bert()
        self.process_data()
        self.init_embedding_model()
        self.train()

# helper –– swap keys and values in a dictionary
def swap_keys_vals(dict_in):
    dict_out = {}
    for key in dict_in.keys():
        val = dict_in[key]
        dict_out[val] = key
    return dict_out


def main():
    annot_model = AnnotEmbedModel()
    annot_model.run()


   # annotator_embeddings = annot_model.get_embs()

if __name__ == '__main__':
    main()
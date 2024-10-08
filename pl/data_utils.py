''' 
dataset needs to be in format:
(transcript, note_k1, note_k2, note_k3)
where the notes are ranked; this is in contrast to pairwise preferences (transcript, note_w, note_l) where w is preferred completion over l
'''
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import json
import condorcet
import pandas as pd
import random

ANNOTS_FNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/data/format_annots.json"
ANNOTS_SCORES_FNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/data/format_annots_scores.json"
NOTES_FOLNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/data/synth_notes/"
TRANSCRIPTS_FOLNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/data/transcripts/"

# create Dataset based object (prep for Dataloader for training policy model)
class PLData(Dataset):
    def __init__(self,
                 annotations_fname=ANNOTS_FNAME,
                 notes_dirname=NOTES_FOLNAME,
                 transcripts_dirname=TRANSCRIPTS_FOLNAME):
        with open(annotations_fname,'r') as inf:
            self.annots = json.load(inf)

        illegal_transcripts = ["anxiety-transcript58","anxiety-transcript436"]
        for ill in illegal_transcripts:
            self.annots.pop(ill)

        # map transcript names to idx-ids to be sure
        self.annots_ids = {}
        for idx, transcript in enumerate(self.annots.keys()):
            self.annots_ids[idx] = transcript
        self.notes_dirname = notes_dirname
        self.transcripts_dirname = transcripts_dirname

    def __len__(self):
        return len(self.annots.keys())

    def __getitem__(self, idx):
        # get transcript
        transcript_name = self.annots_ids[idx]
        with open(self.transcripts_dirname + transcript_name + '.txt','r',encoding='utf8') as inf:
            ret_transcript = ' '.join(inf.readlines())
        # get notes
        with open(self.notes_dirname + transcript_name + '/note0.txt','r',encoding='utf8') as inf:
            ret_note0 = ' '.join(inf.readlines())
        with open(self.notes_dirname + transcript_name + '/note1.txt','r',encoding='utf8') as inf:
            ret_note1 = ' '.join(inf.readlines())
        with open(self.notes_dirname + transcript_name + '/note2.txt','r',encoding='utf8') as inf:
            ret_note2 = ' '.join(inf.readlines())
        # get rankings
        symptom_ranking = combine_rankings(self.annots[transcript_name]['symptom'])
        
        return {'transcript':ret_transcript,
                'note0':ret_note0,
                'note1':ret_note1,
                'note2':ret_note2,
                'rank1':symptom_ranking['1'],
                'rank2':symptom_ranking['2'],
                'rank3':symptom_ranking['3']}

# create score dataset for EVALUATING trained policy model
class PLEvalData(Dataset):
    def __init__(self,
                 annotations_fname=ANNOTS_SCORES_FNAME,
                 notes_dirname=NOTES_FOLNAME,
                 transcripts_dirname=TRANSCRIPTS_FOLNAME):
        with open(annotations_fname,'r') as inf:
            self.annots = json.load(inf)
        # map transcript names to idx-ids to be sure
        self.annots_ids = {}
        for idx, transcript in enumerate(self.annots.keys()):
            self.annots_ids[idx] = transcript
        self.notes_dirname = notes_dirname
        self.transcripts_dirname = transcripts_dirname

    def __len__(self):
        return len(self.annots.keys())

    def __getitem__(self, idx):
        # get transcript
        transcript_name = self.annots_ids[idx]
        with open(self.transcripts_dirname + transcript_name + '.txt','r',encoding='utf8') as inf:
            ret_transcript = ' '.join(inf.readlines())
        # get notes
        with open(self.notes_dirname + transcript_name + '/note0.txt','r',encoding='utf8') as inf:
            ret_note0 = ' '.join(inf.readlines())
        with open(self.notes_dirname + transcript_name + '/note1.txt','r',encoding='utf8') as inf:
            ret_note1 = ' '.join(inf.readlines())
        with open(self.notes_dirname + transcript_name + '/note2.txt','r',encoding='utf8') as inf:
            ret_note2 = ' '.join(inf.readlines())
        # get scores
        scores = combine_scores(self.annots[transcript_name]['symptom'])

        return {'transcript':ret_transcript,
                'note0':ret_note0,
                'note1':ret_note1,
                'note2':ret_note2,
                'score0':scores['note0'],
                'score1':scores['note1'],
                'score2':scores['note2']}

def combine_scores(scores,method="avg"):
    note0 = 0.0
    note1 = 0.0
    note2 = 0.0
    for item in scores:
        note0 += float(item['note0'])
        note1 += float(item['note1'])
        note2 += float(item['note2'])
    note0 /= round(float(len(scores)))
    note1 /= round(float(len(scores)))
    note2 /= round(float(len(scores)))
    return {'note0':note0,'note1':note1,'note2':note2}

# rankings in the format [{"1":"note0",...},...]
def combine_rankings(rankings,method="condorcet"):
    df = pd.DataFrame()
    first_pref, second_pref, third_pref = [], [], []
    for ranking in rankings:
        first_pref.append(ranking["1"])
        second_pref.append(ranking["2"])
        third_pref.append(ranking["3"])
    df['first_preference'] = first_pref
    df['second_preference'] = second_pref
    df['third_preference'] = third_pref

    init_results = condorcet.compute_ranks(df)
    flat_results = [item for sublist in init_results for item in sublist]

    results = {}
    for idx,candidate in enumerate(flat_results):
        results[str(idx + 1)] = candidate

    return results

''' load dataset to dataloader type '''
def load_data(pldata_obj, batch_size, split=.85, seed=123):
    generator = torch.Generator().manual_seed(seed)
    split_train = int(split*len(pldata_obj))
    split_eval = int(len(pldata_obj) - split_train)
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset=pldata_obj,
                                                                lengths=[split_train, split_eval],
                                                                generator=generator)
    return DataLoader(train_dataset,batch_size=batch_size), DataLoader(eval_dataset,batch_size=batch_size)

''' provided dataloader object, view sample '''
def sample_data(_data):
    # sample from data
    item = next(iter(_data))
    print('SAMPLE:')
    print(item)
    return item

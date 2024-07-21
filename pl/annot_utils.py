import json
import pandas as pd

ANNOTATIONS_FNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/annotations/partial_annots_7_18.csv"
ANNOTATIONS_DICT_FNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/annotations/notes_id_maps.json"

def get_rankings(row_obj):
    results = {}

    symptom_rank_note0 = label_to_rank(row_obj[1]['Accuracy of Symptoms\nPlease rank the notes in order of preference, with 1 having the most accurate summary of symptoms, and 3 with the least. [Note 0]'])
    symptom_rank_note1 = label_to_rank(row_obj[1]['Accuracy of Symptoms\nPlease rank the notes in order of preference, with 1 having the most accurate summary of symptoms, and 3 with the least. [Note 1]'])
    symptom_rank_note2 = label_to_rank(row_obj[1]['Accuracy of Symptoms\nPlease rank the notes in order of preference, with 1 having the most accurate summary of symptoms, and 3 with the least. [Note 2]'])
    results['symptom'] = {'note0':symptom_rank_note0,'note1':symptom_rank_note1,'note2':symptom_rank_note2}

    stressor_rank_note0 = label_to_rank(row_obj[1]['Accuracy of stressors\nPlease rank the notes in order of preference, with 1 having the most accurate summary of stressors, and 3 with the least [Note 0]'])
    stressor_rank_note1 = label_to_rank(row_obj[1]['Accuracy of stressors\nPlease rank the notes in order of preference, with 1 having the most accurate summary of stressors, and 3 with the least [Note 1]'])
    stressor_rank_note2 = label_to_rank(row_obj[1]['Accuracy of stressors\nPlease rank the notes in order of preference, with 1 having the most accurate summary of stressors, and 3 with the least [Note 2]'])
    results['stressor'] = {'note0':stressor_rank_note0,'note1':stressor_rank_note1,'note2':stressor_rank_note2}

    return results

def swap_keys_vals(ranks_dict):
    upd_dict = {}
    for cat in ranks_dict.keys():
        upd_dict[cat] = {}
        for key in ranks_dict[cat].keys():
            val = ranks_dict[cat][key]
            upd_dict[cat][val] = key
    return upd_dict

def label_to_rank(label):
    if 'rank 1' in label.lower():
        return 1
    elif 'rank 2' in label.lower():
        return 2
    return 3

def rm_duplicate_annots(df):
    # ensure that all transcript-annotator pairs are unique
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['Transcript ', 'Email Address', 'Timestamp'])
    df = df.drop_duplicates(subset=['Transcript ', 'Email Address'], keep='last')
    return df

# read annotations file and get it in format {transcript: [annot1_ranking, annot2_ranking, ...]}
def read_annots(annots_fname=ANNOTATIONS_FNAME):
    results = {}
    df = pd.read_csv(annots_fname)
    df = rm_duplicate_annots(df)
    print(df.head(5))

    for row in df.iterrows():
        ranks_dict = get_rankings(row)
        ranks_dict = swap_keys_vals(ranks_dict)
        transcript = row[1]['Transcript ']
        if transcript not in results.keys():
            results[transcript] = {'symptom':[ranks_dict['symptom']],
                                   'stressor':[ranks_dict['stressor']]}
        else:
            results[transcript]['symptom'].append(ranks_dict['symptom'])
            results[transcript]['stressor'].append(ranks_dict['stressor'])
    
    print(len(results))      

read_annots()

print('blah')



import json
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(levelname)s - %(message)s')

# reading in
ANNOTATIONS_DICT_FNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/data/notes_id_maps.json"
ANNOTATIONS_FNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/annotations/partial_final.csv"
# writing out
FORMATTED_SCORES_ANNOTATIONS_FNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/data/format_annots_scores.json"
FORMATTED_RANKS_ANNOTATIONS_FNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/data/format_annots_ranks.json"
FORMATTED_SCORES_ANNOTIDS_ANNOTATIONS_FNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/data/format_annots_scores_annotids.json"
FORMATTED_RANKS_ANNOTIDS_ANNOTATIONS_FNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/data/format_annots_ranks_annotids.json"

def get_scores(row_obj):
    results = {}
    symptom_score_note0 = label_to_score(row_obj[1]["\nPlease rate the accuracy of SYMPTOMS in NOTE 0"])
    symptom_score_note1 = label_to_score(row_obj[1]["\nPlease rate the accuracy of SYMPTOMS in NOTE 1"])
    symptom_score_note2 = label_to_score(row_obj[1]["\nPlease rate the accuracy of SYMPTOMS in NOTE 2"])
    results['symptom'] = {'note0':symptom_score_note0,'note1':symptom_score_note1,'note2':symptom_score_note2}

    stressor_score_note0 = label_to_score(row_obj[1]["\nPlease rate the accuracy of STRESSORS in NOTE 0"])
    stressor_score_note1 = label_to_score(row_obj[1]["\nPlease rate the accuracy of STRESSORS in NOTE 1"])
    stressor_score_note2 = label_to_score(row_obj[1]["\nPlease rate the accuracy of STRESSORS in NOTE 2"])
    results['stressor'] = {'note0':stressor_score_note0,'note1':stressor_score_note1,'note2':stressor_score_note2}

    acc_score_note0 = label_to_score(row_obj[1]["\nPlease rate the OVERALL ACCURACY of NOTE 0"])
    acc_score_note1 = label_to_score(row_obj[1]["\nPlease rate the OVERALL ACCURACY of NOTE 1"])
    acc_score_note2 = label_to_score(row_obj[1]["\nPlease rate the OVERALL ACCURACY of NOTE 2"])
    results['accuracy'] = {'note0':acc_score_note0,'note1':acc_score_note1,'note2':acc_score_note2}

    readab_score_note0 = label_to_score(row_obj[1]["\nPlease rate the READABILITY OF NOTE 0"])
    readab_score_note1 = label_to_score(row_obj[1]["\nPlease rate the READABILITY OF NOTE 1"])
    readab_score_note2 = label_to_score(row_obj[1]["\nPlease rate the READABILITY OF NOTE 2"])
    results['readability'] = {'note0':readab_score_note0,'note1':readab_score_note1,'note2':readab_score_note2}

    return results

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

    acc_rank_note0 = label_to_rank(row_obj[1]['Overall Accuracy\nPlease rank the notes in order of preference, with 1 having the most accurate summary of the session overall, and 3 with the least [Note 0]'])
    acc_rank_note1 = label_to_rank(row_obj[1]['Overall Accuracy\nPlease rank the notes in order of preference, with 1 having the most accurate summary of the session overall, and 3 with the least [Note 1]'])
    acc_rank_note2 = label_to_rank(row_obj[1]['Overall Accuracy\nPlease rank the notes in order of preference, with 1 having the most accurate summary of the session overall, and 3 with the least [Note 2]'])
    results['accuracy'] = {'note0':acc_rank_note0,'note1':acc_rank_note1,'note2':acc_rank_note2}

    readab_rank_note0 = label_to_rank(row_obj[1]['Readability\nPlease rank the notes in order of preference, with 1 having the most readable note, and 3 with the least [Note 0]'])
    readab_rank_note1 = label_to_rank(row_obj[1]['Readability\nPlease rank the notes in order of preference, with 1 having the most readable note, and 3 with the least [Note 1]'])
    readab_rank_note2 = label_to_rank(row_obj[1]['Readability\nPlease rank the notes in order of preference, with 1 having the most readable note, and 3 with the least [Note 2]'])
    results['readability'] = {'note0':readab_rank_note0,'note1':readab_rank_note1,'note2':readab_rank_note2}

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

def label_to_score(label):
    if '1.' in label:
        return 1
    elif '2.' in label:
        return 2
    elif '3.' in label:
        return 3
    return 4

def rm_duplicate_annots(df):
    # ensure that all transcript-annotator pairs are unique
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['Transcript ', 'Email Address', 'Timestamp'])
    df = df.drop_duplicates(subset=['Transcript ', 'Email Address'], keep='last')
    return df

# read annotations file and get it in format {transcript: [annot1_ranking, annot2_ranking, ...]}
def read_annots_ranks(annots_fname=ANNOTATIONS_FNAME,
                      group="general"):
    results = {}
    df = pd.read_csv(annots_fname)
    df = rm_duplicate_annots(df)

    for row in df.iterrows():
        ranks_dict = get_rankings(row)
        ranks_dict = swap_keys_vals(ranks_dict)
        transcript = row[1]['Transcript ']
        annot_id = row[1]['Email Address']
        # schizophrenia is misspelled in the annotator form
        if 'schizphrenia' in transcript:
            transcript = transcript.replace('schizphrenia','schizophrenia')

        if group == "general":
            if transcript not in results.keys():
                results[transcript] = {'symptom':[ranks_dict['symptom']],
                                    'stressor':[ranks_dict['stressor']],
                                    'accuracy':[ranks_dict['accuracy']],
                                    'readability':[ranks_dict['readability']]}
            else:
                results[transcript]['symptom'].append(ranks_dict['symptom'])
                results[transcript]['stressor'].append(ranks_dict['stressor'])
                results[transcript]['accuracy'].append(ranks_dict['accuracy'])
                results[transcript]['readability'].append(ranks_dict['readability'])
        else:
            if annot_id not in results.keys():
                results[annot_id] = {transcript:{'symptom':[ranks_dict['symptom']],
                                                    'stressor':[ranks_dict['stressor']],
                                                    'accuracy':[ranks_dict['accuracy']],
                                                    'readability':[ranks_dict['readability']]}}
            else:
                results[annot_id][transcript] = {'symptom':[ranks_dict['symptom']],
                                                    'stressor':[ranks_dict['stressor']],
                                                    'accuracy':[ranks_dict['accuracy']],
                                                    'readability':[ranks_dict['readability']]}


    return results

# read annotations file and get it in format {transcript: [annot1_ranking, annot2_ranking, ...]}
def read_annots_scores(annots_fname=ANNOTATIONS_FNAME,
                       group="general"): # general, annotator // general -- no annot ids, annotator -- group by annotator
    results = {}
    df = pd.read_csv(annots_fname)
    df = rm_duplicate_annots(df)

    for row in df.iterrows():
        scores_dict = get_scores(row)
        transcript = row[1]['Transcript ']
        annot_id = row[1]['Email Address']
        # schizophrenia is misspelled in the annotator form
        if 'schizphrenia' in transcript:
            transcript = transcript.replace('schizphrenia','schizophrenia')
        # if general, group by transcript
        if group == "general":
            if transcript not in results.keys():
                results[transcript] = {'symptom':[scores_dict['symptom']],
                                    'stressor':[scores_dict['stressor']],
                                    'accuracy':[scores_dict['accuracy']],
                                    'readability':[scores_dict['readability']]}
            else:
                results[transcript]['symptom'].append(scores_dict['symptom'])
                results[transcript]['stressor'].append(scores_dict['stressor'])
                results[transcript]['accuracy'].append(scores_dict['accuracy'])
                results[transcript]['readability'].append(scores_dict['readability'])
        # group by annotator
        else:
            if annot_id not in results.keys():
                results[annot_id] = {transcript:{'symptom':[scores_dict['symptom']],
                                                    'stressor':[scores_dict['stressor']],
                                                    'accuracy':[scores_dict['accuracy']],
                                                    'readability':[scores_dict['readability']]}}
            else:
                results[annot_id][transcript] = {'symptom':[scores_dict['symptom']],
                                                 'stressor':[scores_dict['stressor']],
                                                 'accuracy':[scores_dict['accuracy']],
                                                 'readability':[scores_dict['readability']]}

    return results

def format_annotator_groups():
    scores = read_annots_scores()
    ranks = read_annots_ranks()


def main():
    logging.info("Processing and formatting annotations...")
    format_annots_scores = read_annots_scores(group="general")
    format_annots_ranks = read_annots_ranks(group="general")
    import json
    logging.info("Writing formatted annotations...")
    with open(FORMATTED_SCORES_ANNOTATIONS_FNAME,'w+') as outf:
        json.dump(format_annots_scores,outf)
    with open(FORMATTED_RANKS_ANNOTATIONS_FNAME,'w+') as outf:
        json.dump(format_annots_ranks,outf)

    logging.info("Processing and formatting annotations ANNOTATOR GROUPS...")
    format_annots_scores = read_annots_scores(group="annotator")
    format_annots_ranks = read_annots_ranks(group="annotator")
    import json
    logging.info("Writing formatted annotations ANNOTATOR GROUPS...")
    with open(FORMATTED_SCORES_ANNOTIDS_ANNOTATIONS_FNAME,'w+') as outf:
        json.dump(format_annots_scores,outf)
    with open(FORMATTED_RANKS_ANNOTIDS_ANNOTATIONS_FNAME,'w+') as outf:
        json.dump(format_annots_ranks,outf)

if __name__ == '__main__':
    main()

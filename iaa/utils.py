
import json

ANNOTS_IDS_FNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/data/format_annots_with_ids.json"
ANNOTS_FNAME = "/home/agunal/scratch/goldkind-clinical-ai/tmpdir/data/format_annots.json"

''' 
for iaa, only need annots per transcript
'''
def read_for_iaa(annots_fname=ANNOTS_FNAME):
    with open(annots_fname,'r') as inf:
        ret_dict = json.load(inf)
    return ret_dict

'''
for annotator clustering, need IDs for each annotation
'''
def read_for_annot_analysis(annots_fname=ANNOTS_IDS_FNAME):
    with open(annots_fname,'r') as inf:
        ret_dict = json.load(inf)
    return ret_dict


''' 
get annotator statistics 
'''
def annot_stats():
    stats_dict = {'num_uniq_annotators':0,
                  'min_annotators':100000, # min number of annotators on a transcript 
                  'max_annotators':0, # max number of annotators on a transcript
                  'avg_annotators':0.0, # avg number of annotators on a transcript
                  'total_transcripts':0} # num transcripts w/ at least 1 annotation
    
    annots_with_ids = read_for_annot_analysis()
    track_transcripts = {}
    for annot_id in annots_with_ids.keys():
        # update transcript-specific counts
        for transcript_id in annots_with_ids[annot_id].keys():
            if transcript_id not in track_transcripts.keys():
                track_transcripts[transcript_id] = {'num_annots':1}
            else:
                track_transcripts[transcript_id]['num_annots'] += 1

    for transcript_id in track_transcripts.keys():
        num_annotators = len(track_transcripts[transcript_id].keys())
        if num_annotators < stats_dict['min_annotators']:
            stats_dict['min_annotators'] = num_annotators
        if num_annotators > stats_dict['max_annotators']:
            stats_dict['max_annotators'] = num_annotators
        
        stats_dict['avg_annotators'] += num_annotators

    stats_dict['num_uniq_annotators'] = len(annots_with_ids.keys())
    stats_dict['avg_annotators'] /= stats_dict['num_uniq_annotators']
    stats_dict['total_transcripts'] = len(track_transcripts.keys())

    return stats_dict










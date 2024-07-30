
import statsmodels

from utils import *

'''
for scores, use fleiss kappa + % agreement
'''
def score_iaa():
    annots = read_scores_for_iaa()
    fleiss_input = [] # rows = annotators, columns = category
    statsmodels.stats.inter_rater.fleiss_kappa(table, method='fleiss')
    return













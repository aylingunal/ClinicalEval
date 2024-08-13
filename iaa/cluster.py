
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

annot_embs_fname = "/home/agunal/scratch/goldkind-clinical-ai/dev/ClinicalEval/iaa/annot_embeds.parquet"

# load the embeddings
def read_annot_embs():
    df = pd.read_parquet(annot_embs_fname)
    annotator_ids = df['annotator_id']
    embeddings = df['embedding']
    return annotator_ids, embeddings

def similarities():
    annotator_ids, embeddings = read_annot_embs()
    embeddings_ = np.stack(embeddings)
    cosine_sim_matrix = cosine_similarity(embeddings_)
    print(cosine_sim_matrix)
    return annotator_ids, embeddings_, cosine_sim_matrix

def clusters():
    annotator_ids, _, cosine_sim_matrix = similarities()
    # convert similarity matrix to distance matrix
    distance_matrix = 1 - cosine_sim_matrix
    condensed_distance_matrix = squareform(distance_matrix, checks=False)
    # hierarchical clustering
    Z = linkage(condensed_distance_matrix, method='average')
    max_d = 1 - 0.85
    cluster_labels = fcluster(Z, max_d, criterion='distance')
    # group annotator IDs by clusters
    from collections import defaultdict
    clusters_dict = defaultdict(list)
    for annotator_id, cluster_id in zip(annotator_ids, cluster_labels):
        clusters_dict[cluster_id].append(annotator_id)
    # convert the clusters_dict to a list of clusters
    clusters = list(clusters_dict.values())
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")


def similarities_heatmap():
    annotator_ids, embeddings = read_annot_embs()
    # # mapping from original names to numbers
    # id_mapping = {name: f'ID{i+1}' for i, name in enumerate(annotator_ids)}
    # # anonymize ids
    # anonymized_data = {id_mapping[name]: name for name in annotator_ids}
    embeddings_ = np.stack(embeddings)
    cosine_sim_matrix = cosine_similarity(embeddings_)
    #cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=anonymized_data.keys(), columns=anonymized_data.keys())
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=annotator_ids, columns=annotator_ids)
    # plot
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cosine_sim_df, annot=True, cmap='coolwarm')#, vmin=-1, vmax=1, center=0, cbar_kws={'shrink': .8})
    ax.figure.tight_layout()
    plt.title('Cosine Similarity Heatmap')
    plt.savefig("annotator_similarities_heatmap.png")


def main():
    clusters()

if __name__=='__main__':
    main()







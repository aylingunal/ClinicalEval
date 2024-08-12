
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

annot_embs_fname = "/home/agunal/scratch/goldkind-clinical-ai/dev/ClinicalEval/iaa/annot_embeds.parquet"

# load the embeddings
def read_annot_embs():
    df = pd.read_parquet(annot_embs_fname)
    annotator_ids = df['annotator_id']
    embeddings = df['embedding']
    return annotator_ids, embeddings

def similarities():
    annotator_ids, embeddings = read_annot_embs()

    # # Create a mapping from original names to numbers
    # id_mapping = {name: f'ID{i+1}' for i, name in enumerate(annotator_ids)}
    # # Anonymize IDs
    # anonymized_data = {id_mapping[name]: name for name in annotator_ids}

    embeddings_ = np.stack(embeddings)
    cosine_sim_matrix = cosine_similarity(embeddings_)
    #cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=anonymized_data.keys(), columns=anonymized_data.keys())
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=annotator_ids, columns=annotator_ids)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cosine_sim_df, annot=True, cmap='coolwarm')#, vmin=-1, vmax=1, center=0, cbar_kws={'shrink': .8})
    ax.figure.tight_layout()
    plt.title('Cosine Similarity Heatmap')
    plt.savefig("annotator_similarities_heatmap.png")

def clusters(threshold=.85):
    return

def main():
    similarities()

if __name__=='__main__':
    main()







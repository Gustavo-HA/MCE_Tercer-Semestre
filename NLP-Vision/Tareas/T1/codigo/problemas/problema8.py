import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

def doc2vec(doc, w2v):
    """Takes tokenized document to vector representation."""
    word_vectors = np.array([w2v.wv[word] for word in doc if word in w2v.wv])
    return word_vectors.mean(axis=0)

def main(data:pd.DataFrame, doc_tokens, w2v):

    doc_vectors = np.array([doc2vec(doc, w2v) for doc in doc_tokens])

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(doc_vectors)
    
    centroides = kmeans.cluster_centers_

    dist_centroides = np.zeros((5000,5))
    for i, doc_vec in enumerate(doc_vectors):
        for j in range(5):
            dist_centroides[i, j] = np.linalg.norm(doc_vec - centroides[j])

    doc_mas_cercanos = np.argmin(dist_centroides, axis=0)

    print("Documentos mas cercanos a cada centroide:")
    for i, doc in enumerate(doc_mas_cercanos):
        print(f"\n- Centroide {i}\nPolaridad: {data.loc[doc, 'Polarity']}\nDocumento:\n{data.loc[doc, 'Review']}")
    
    return doc_vectors

if __name__ == "__main__":
    from codigo.problemas.problema1 import main as p1
    from codigo.problemas.problema7 import main as p7
    from codigo.config import DATA_FILE

    tokenizer, spanish_stopwords, fdist, data, documentos, doc_tokens = p1(DATA_FILE, print_results=False)
    w2v = p7(doc_tokens, print_results=False)
    main(data, doc_tokens, w2v)
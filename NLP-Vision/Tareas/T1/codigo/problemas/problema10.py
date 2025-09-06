from collections import Counter
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def construir_vocab(corpus_tokens, min_count=1):
    """
    Construye vocabulario a partir de una lista de listas de tokens.
    min_count: frecuencia mínima para incluir palabra en el vocabulario.
    """
    freqs = Counter()
    for tokens in corpus_tokens:
        freqs.update(tokens)
    vocab = [w for w, c in freqs.items() if c >= min_count]
    vocab.sort()  # orden estable
    word2id = {w: i for i, w in enumerate(vocab)}
    return vocab, word2id

def matriz_coocurrencia(corpus, tokenizer, window_size=2, min_count=1, directed=True):
    """
    Construye la matriz W (palabra objetivo x palabra contexto).
    - window_size: tamaño de la ventana simétrica alrededor del target.
    - directed=True: solo cuenta (i -> j). Si False, también suma (j -> i).
    Devuelve: (W_csr, vocab, word2id)
    """
    # Tokenizar corpus
    corpus_tokens = [tokenizer.tokenize(doc) for doc in corpus]
    # Vocabulario
    vocab, word2id = construir_vocab(corpus_tokens, min_count=min_count)
    V = len(vocab)

    rows, cols, data = [], [], []

    for tokens in corpus_tokens:
        ids = [word2id[w] for w in tokens if w in word2id]
        n = len(ids)
        for i, wi in enumerate(ids):
            # ventana [i-window_size, i+window_size], excluyendo i
            left = max(0, i - window_size)
            right = min(n, i + window_size + 1)
            for j in range(left, right):
                if i == j:
                    continue
                wj = ids[j]
                # cuenta ocurrencia en el contexto
                rows.append(wi)
                cols.append(wj)
                data.append(1.0)
                # si quieres matriz estrictamente simétrica, puedes sumar también el espejo
                if not directed:
                    rows.append(wj)
                    cols.append(wi)
                    data.append(1.0)

    if not rows:
        # corpus vacío o vocabulario filtrado demasiado estricto
        W = csr_matrix((V, V), dtype=np.float64)
    else:
        W = coo_matrix((np.array(data, dtype=np.float64),
                        (np.array(rows), np.array(cols))), shape=(V, V)).tocsr()

    return W, vocab, word2id

def lsa_embeddings(W, k=10, random_state=0):
    """
    Aplica SVD truncado a W:
        W ≈ U_k Σ_k V_k^T
    TruncatedSVD.fit_transform(W) ≈ U_k Σ_k  (embeddings de filas/palabras objetivo)
    Devuelve: X (V x k) con embeddings densos.
    """
    k = min(k, min(W.shape) - 1)  # seguridad
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    X = svd.fit_transform(W)  # U_k Σ_k
    return X, svd

def main(tokenizer, documentos):
    """
    Flujo principal para cargar datos, preprocesar, construir W, obtener embeddings LSA
    e imprimir análisis de tópicos.
    """
    # Preprocesamiento: tokenización y construcción de vocabulario
    corpus_tokens = [tokenizer.tokenize(doc) for doc in documentos]
    vocab, word2id = construir_vocab(corpus_tokens, min_count=2)

    # Construcción de la matriz de coocurrencia
    W, vocab, word2id = matriz_coocurrencia(documentos, tokenizer=tokenizer, window_size=2, min_count=2, directed=True)

    # LSA vía SVD truncado con 50 tópicos
    X, svd = lsa_embeddings(W, k=50, random_state=42)

    # Mostrar los términos más relevantes por tópico
    n_top_words = 10
    terms = np.array(vocab)
    for topic_idx, component in enumerate(svd.components_):
        top_indices = np.argsort(component)[::-1][:n_top_words]
        top_terms = terms[top_indices]
        print(f"Tópico {topic_idx+1}: {', '.join(top_terms)}")

    # Identificar tópicos más informativos según varianza explicada
    explained_var = svd.explained_variance_ratio_
    top_topics = np.argsort(explained_var)[::-1][:5]
    print("\nTópicos más informativos (por varianza explicada):")
    for idx in top_topics:
        print(f"Tópico {idx+1}: varianza explicada = {explained_var[idx]:.4f}")

    # Analizar coherencia: correlación entre términos principales de cada tópico
    print("\nCoherencia de tópicos:")
    coherencias = []
    for topic_idx, component in enumerate(svd.components_):
        top_indices = np.argsort(component)[::-1][:n_top_words]
        vectors = X[top_indices]
        sim = cosine_similarity(vectors)
        upper_tri = sim[np.triu_indices_from(sim, k=1)]
        coherencia = upper_tri.mean() if len(upper_tri) > 0 else 0
        coherencias.append(coherencia)
        print(f"Coherencia del tópico {topic_idx+1}: {coherencia:.3f}")
        
    # Topicos con mayor coherencia
    
    top_coherencias = np.argsort(coherencias)[::-1][:5]
    print("\nTópicos con mayor coherencia:")
    for idx in top_coherencias:
        print(f"Tópico {idx+1}: coherencia = {coherencias[idx]:.4f}")
    

if __name__ == "__main__":
    from codigo.problemas.problema1 import main as p1
    from codigo.problemas.problema3 import main as p3
    from codigo.config import DATA_FILE

    tokenizer, spanish_stopwords, fdist, data, documentos, doc_tokens = p1(DATA_FILE, print_results=False)
    data_normalizada = p3(data, tokenizer, spanish_stopwords, savefig=False)
    main(tokenizer, data_normalizada["Review"].tolist())
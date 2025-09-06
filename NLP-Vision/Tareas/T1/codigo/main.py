from codigo.config import DATA_FILE
import pandas as pd

from codigo.problemas import (
    problema1,
    problema2,
    problema3,
    problema4,
    problema5_6,
    problema7,
    problema8,
    problema9,
    problema10
)

def run_all():
    """
    Ejecuta todos los problemas en secuencia.
    """
    print("--- Problema 1: Descripción del Corpus ---")
    tokenizer, spanish_stopwords, fdist, data, documentos, doc_tokens = problema1.main(DATA_FILE, print_results=True)

    print("\n--- Problema 2: Ley de Zipf ---")
    problema2.main(fdist, savefig=True)
    
    print("\n--- Problema 3: Palabras importantes por clase ---")
    data_normalizada = problema3.main(data, tokenizer, spanish_stopwords, savefig=True)

    print("\n--- Problema 4: Patrones gramaticales (POS tagging) ---")
    problema4.main(data)
    
    print("\n--- Problema 5 & 6: Representaciones BoW y Bigramas ---")
    problema5_6.main(data)
    
    print("\n--- Problema 7: Word2Vec y analogías ---")
    w2v = problema7.main(doc_tokens, print_results=True)

    print("\n--- Problema 8: Embeddings de documento y clusterización ---")
    doc_vectors = problema8.main(data, doc_tokens, w2v)
    
    print("\n--- Problema 9: Clasificación ---")
    problema9.main(data, documentos, doc_vectors, tokenizer, spanish_stopwords)
    
    print("\n--- Problema 10: LSA con 50 tópicos ---")
    problema10.main(tokenizer, data_normalizada["Review"].tolist())

if __name__ == "__main__":
    run_all()


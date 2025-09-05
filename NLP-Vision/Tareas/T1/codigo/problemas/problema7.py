from gensim.models import Word2Vec

def get_analogies(model: Word2Vec, positive, negative):
    """
    Obtiene las palabras más similares a una palabra dada, utilizando un modelo de Word2Vec.
    """
    return model.wv.most_similar(positive=positive, negative=negative)

def main(doc_tokens, print_results=True):
    model = Word2Vec(sentences=doc_tokens, vector_size=75, window=2, min_count=1, workers=4, sg=1, seed=1)

    examples = [
        (['habitacion', 'restaurante'], ['hotel']),
        (['mesa', 'hotel'], ['restaurante']),
        (['deliciosa', 'platillo'], None),
        (['servicio', 'comida'], ['atención']),
        (['comida', 'servicio'], ['platillo'])
    ]
    
    if print_results:
        for positive, negative in examples:
            analogies = get_analogies(model, positive=positive, negative=negative)
            print(f"Positivo: {positive}, Negativo: {negative}\nResultado: {analogies[:4]}\n")
        
    return model

if __name__ == "__main__":
    from codigo.problemas.problema1 import main as p1
    from codigo.config import DATA_FILE

    tokenizer, spanish_stopwords, fdist, data, documentos, doc_tokens = p1(DATA_FILE, print_results=False)
    main(doc_tokens, print_results=True)
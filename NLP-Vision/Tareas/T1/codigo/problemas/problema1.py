from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist
from nltk.corpus import stopwords
import pandas as pd

from ..config import DATA_FILE

def main(data_path, is_test=False):
    data = pd.read_csv(data_path)
    data = data.drop(columns=["Town","Region","Type"])
    
    documentos = data["Review"].tolist()
    corpus = " ".join(documentos)
    n = len(documentos)
    
    
    # Punto 1
    tokenizer = RegexpTokenizer(r'[a-zA-ZáéíóúñÁÉÍÓÚÑ]+')
    doc_tokens = [tokenizer.tokenize(doc) for doc in documentos]
    tokens = [token for token in tokenizer.tokenize(corpus)]
    vocab = set(tokens)
    if not is_test:    
        print(f"Numero de documentos: {n}")
        print(f"Numero de tokens en el corpus: {len(tokens)}")
        print(f"Numero de palabras en el vocabulario: {len(vocab)}")
    
    # Punto 2
    
    fdist = FreqDist(tokens)
    hapax_legomena = [word for word, freq in fdist.items() if freq == 1]
    if not is_test:
        print(f"Numero de hapax legomena: {len(hapax_legomena)}")
        print(f"Proporcion de hapax legomena con el vocabulario: {len(hapax_legomena) / len(vocab) * 100 if vocab else 0:.3f}%")

    # Punto 3

    spanish_stopwords = set(stopwords.words('spanish'))
    contador_stopwords = 0
    stopwords_corpus = set()
    for token in tokens:
        if token.lower() in spanish_stopwords:
            contador_stopwords += 1
            stopwords_corpus.add(token.lower())
    if not is_test:
        print(f"Numero de stopwords: {contador_stopwords}")
        print(f"Numero de stopwords en el vocabulario: {len(stopwords_corpus)}")
        print(f"Porcentaje de stopwords: {contador_stopwords / len(tokens) * 100 if tokens else 0:.3f}%")
    
    ## Parte 4. Estadisticas por clase
    for polarity in sorted(data['Polarity'].unique()):
        df_reducido = data.loc[data['Polarity'] == polarity, 'Review'].tolist()
        corpus_reducido = ' '.join(df_reducido)
        n = len(df_reducido)
        tokens_r = tokenizer.tokenize(corpus_reducido)
        vocab_r = set(tokens_r)
        
        if not is_test:
            print(f"Estadisticas para la polaridad {polarity}:")
            print(f" - Cantidad de textos: {n}")
            print(f" - Cantidad de palabras: {len(tokens_r)}")
            print(f" - Cantidad de palabras en el vocabulario: {len(vocab_r)}")
        
    return tokenizer, spanish_stopwords, fdist, data, documentos, doc_tokens
    
    
if __name__ == "__main__":
    main(DATA_FILE)

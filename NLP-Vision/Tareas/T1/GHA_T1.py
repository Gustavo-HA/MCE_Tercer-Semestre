import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

class Tarea1:
    def __init__(self, data_filename):
        self.data = pd.read_csv(data_filename)

    def no1(self):
        """Descripcion del Corpus"""
        
        # Punto 1
        documentos = self.data["Review"].tolist()
        n = len(documentos)
        tokenizer = RegexpTokenizer(r'\w+')
        doc_tokens = [tokenizer.tokenize(doc) for doc in documentos]
        tokens = [token for doc in doc_tokens for token in doc]
        vocab = set(tokens)
        print(f"Numero de documentos: {n}")
        print(f"Numero de tokens en el corpus: {len(tokens)}")
        print(f"Numero de palabras en el vocabulario: {len(vocab)}")

        # Punto 2
        # Hapax legomena y su proporcion.
        hapax_legomena = [word for word in vocab if tokens.count(word) == 1]
        print(f"Numero de hapax legomena: {len(hapax_legomena)}")
        print(f"Proporcion de hapax legomena con el vocabulario: {len(hapax_legomena) / len(vocab) * 100 if vocab else 0:.5f}")
        
        # Punto 3
        # Porcentaje de stopwords

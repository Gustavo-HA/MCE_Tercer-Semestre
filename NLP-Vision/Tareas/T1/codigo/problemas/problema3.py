import nltk
from nltk import FreqDist
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def normalizar_texto(texto: str, 
                     tokenizador: nltk.RegexpTokenizer,
                     stopwords: set) -> str:
    """
    Minusculas, tokenizado palabras, quitar stopwords.
    """
    tokens = tokenizador.tokenize(texto.lower())
    tokens_normalizados = [token for token in tokens if token not in stopwords]
    return ' '.join(tokens_normalizados)

def normalizar_reviews(data: pd.DataFrame, 
                       tokenizador: nltk.RegexpTokenizer,
                       stopwords: set) -> pd.DataFrame:
    """
    Normalizar todas las reviews en el DataFrame.
    """
    data_normalizada = data.copy()
    data_normalizada['Review'] = data_normalizada['Review'].apply(
        lambda texto: normalizar_texto(texto, tokenizador, stopwords)
    )
    return data_normalizada

def graficar_freqs(data_normalizada: pd.DataFrame, 
                   tokenizador: nltk.RegexpTokenizer,
                   savefig: bool = False):
    
    sns.set_theme(font_scale=1.6)

    for clase in sorted(data_normalizada["Polarity"].unique()):
        corpus_r = data_normalizada.loc[data_normalizada["Polarity"] == clase,"Review"].tolist()
        
        tokens_r = [word for review in corpus_r for word in tokenizador.tokenize(review)]
        freq = FreqDist(tokens_r)

        palabra_array = []
        frecuencia_array = []
        for palabra, frecuencia in freq.most_common(10):
            palabra_array.append(palabra)
            frecuencia_array.append(frecuencia)

        sns.barplot(x=frecuencia_array, y=palabra_array, orient="h")

        if savefig:
            plt.savefig(f"./figures/palabras_comunes_clase_{clase:.0f}.pdf", bbox_inches='tight', dpi=400)


def main(data: pd.DataFrame, 
         tokenizador: nltk.RegexpTokenizer,
         stopwords: set,
         savefig: bool = False):
    data_normalizada = normalizar_reviews(data, tokenizador, stopwords)
    graficar_freqs(data_normalizada, tokenizador, savefig)
    return data_normalizada

if __name__ == "__main__":
    from problema1 import main as p1
    from ..config import DATA_FILE

    tokenizer, spanish_stopwords, fdist, data, documentos, doc_tokens = p1(DATA_FILE)
    main(data, tokenizer, spanish_stopwords, savefig=True)
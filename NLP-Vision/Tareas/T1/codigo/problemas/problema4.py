import spacy
import pandas as pd
from itertools import chain
from collections import Counter

def aplicar_pos_tagging(texto: str, nlp) -> list:
    """
    Aplica POS tagging a un texto usando el modelo de spaCy.
    """
    doc = nlp(texto)
    return " ".join([token.pos_ for token in doc])


def get_ngrams(text, n=4):
    """
    Toma una cadena de texto, la divide en tokens y devuelve una lista de n-gramas.
    """
    tokens = text.split()
    if len(tokens) < n:
        return []
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def get_tetragramas(data: pd.DataFrame, nlp) -> dict:
    """
    Obtiene los tetragramas de un texto dado.
    """
    class_counts = {}
    for cls in sorted(data['Polarity'].unique()):
        class_df = data[data['Polarity'] == cls]

        all_tetragrams = list(chain.from_iterable(class_df['tetragram']))
        counter = Counter(all_tetragrams)
        class_counts[cls] = counter.most_common(5)

    return class_counts

def main(data: pd.DataFrame) -> pd.DataFrame:
    nlp = spacy.load("es_core_news_sm")

    data['pos'] = data['Review'].apply(
                lambda texto: aplicar_pos_tagging(texto, nlp)
    )

    data['tetragram'] = data['pos'].apply(lambda x: get_ngrams(x, n=4))

    tetragramas = get_tetragramas(data, nlp)
    
    for cls, grams in tetragramas.items():
        print(f"\nLos 5 tetragramas más comunes para la clase {cls:.0f} son:")
        for tetragram, count in grams:
            print(f"- '{tetragram}' : {count} veces")

    data.drop(columns=['pos', 'tetragram'], inplace=True)

    return data

if __name__ == "__main__":
    from codigo.config import DATA_FILE
    
    data = pd.read_csv(DATA_FILE)
    data = data.drop(columns=["Town","Region","Type"])

    main(data)
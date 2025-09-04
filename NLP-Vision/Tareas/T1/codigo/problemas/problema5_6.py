from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
import pandas as pd
from sklearn.feature_selection import (
    chi2,
    SelectKBest
)
import nltk

def get_representations(data: pd.DataFrame, count_args={}, tfidf_args={}):
    """
    Obtiene las representaciones Bag of Words y TF-IDF del texto.
    """
    vectorizer_bow = CountVectorizer(**count_args)
    X_bow = vectorizer_bow.fit_transform(data['Review']).toarray()

    vectorizer_tfidf = TfidfVectorizer(**tfidf_args)
    X_tfidf = vectorizer_tfidf.fit_transform(data['Review']).toarray()

    tfidf_features = vectorizer_tfidf.get_feature_names_out()
    bow_features = vectorizer_bow.get_feature_names_out()

    y = data['Polarity'].values

    return X_bow, X_tfidf, bow_features, tfidf_features, y

def get_k_best_features(X, y, features, k=20):
    selector = SelectKBest(score_func=chi2, k=k)
    X_new = selector.fit_transform(X, y) # noqa: F841
    mask = selector.get_support(indices=True)

    selected_features = [features[i] for i in mask]

    return selected_features

def p5(data: pd.DataFrame, k=20, spanish_stopwords=set(nltk.corpus.stopwords.words('spanish'))):
    X_bow, X_tfidf, bow_features, tfidf_features, y = get_representations(
        data
    )
    
    _, bow_kbest_features = get_k_best_features(X_bow, y, bow_features, k)
    _, tfidf_kbest_features = get_k_best_features(X_tfidf, y, tfidf_features, k)

    print(f"Top {k} características más importantes según Chi-cuadrado (TF):")
    for feature in bow_kbest_features:
        print(f"- {feature}")

    print(f"\nTop {k} características más importantes según Chi-cuadrado (TF-IDF):")
    for feature in tfidf_kbest_features:
        print(f"- {feature}")
        
    contador_tf = 0
    contador_tfidf = 0

    for i in range(20):
        if bow_kbest_features[i] in spanish_stopwords:
            contador_tf += 1
        if tfidf_kbest_features[i] in spanish_stopwords:
            contador_tfidf += 1

    print(f"Stopwords TF: {contador_tf}")
    print(f"Stopwords TF-IDF: {contador_tfidf}")

    return 0

def p6(data: pd.DataFrame, k=20, spanish_stopwords=set(nltk.corpus.stopwords.words('spanish'))):

    tfidf_args = {"ngram_range": (2, 2)}
    count_args = {"ngram_range": (2, 2)}

    X_bow, X_tfidf, bow_features, tfidf_features, y = get_representations(
        data,
        count_args=count_args, 
        tfidf_args=tfidf_args
    )

    bow_kbest_features = get_k_best_features(X_bow, y, bow_features, k)
    tfidf_kbest_features = get_k_best_features(X_tfidf, y, tfidf_features, k)

    common_features = set(bow_kbest_features).intersection(set(tfidf_kbest_features))


    # Imprimir interseccion
    print(f"Características comunes en ambas representaciones (Top {k}):")
    for feature in common_features:
        print(f"- {feature}")


    # Contar stopwords
    contador_tf = 0
    contador_tfidf = 0
    for i in range(20):
        if bow_kbest_features[i].split()[0] in spanish_stopwords:
            contador_tf += 1
        if bow_kbest_features[i].split()[1] in spanish_stopwords:
            contador_tf += 1
        if tfidf_kbest_features[i].split()[0] in spanish_stopwords:
            contador_tfidf += 1
        if tfidf_kbest_features[i].split()[1] in spanish_stopwords:
            contador_tfidf += 1

    print(f"Stopwords TF: {contador_tf}")
    print(f"Stopwords TF-IDF: {contador_tfidf}")

    return 0


def main(data: pd.DataFrame, k=20):
    print("\nProblema 5:")
    p5(data, k)
    print("\nProblema 6:")
    p6(data, k)
    return 0

if __name__ == "__main__":
    from ..config import DATA_FILE
    
    data = pd.read_csv(DATA_FILE)
    
    main(data, k=20)
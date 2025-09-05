from sklearn import svm
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from codigo.problemas.problema8 import doc2vec
from nltk.stem import SnowballStemmer
from nltk import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns

def evaluar_representacion(X_train, X_test, y_train, y_test, inciso=""):
    
    parameters = {"C": [.05, .12, .25, .5, 1, 2, 4]}
    
    svr = svm.LinearSVC(class_weight='balanced', max_iter=10000)
    grid = GridSearchCV(estimator=svr, param_grid=parameters,
                        n_jobs=6, scoring="f1_macro", cv=5)
    grid.fit(X_train, y_train)
    
    y_pred = grid.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    class_report = "\n" + classification_report(y_test, y_pred, digits=3)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(1, 6), yticklabels=range(1, 6))
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./figures/confusion_matrix_{inciso}.pdf", bbox_inches='tight')
    plt.clf()

    return {
        "confusion_matrix": f"./figures/confusion_matrix_{inciso}.pdf",
        "classification_report": class_report
    }

def reporte_metricas(x_train_index, x_test_index,
                     y_train, y_test,
                     vec_list, print_results=True, inciso=""):
    X_train = vec_list[x_train_index]
    X_test = vec_list[x_test_index]
    metricas = evaluar_representacion(X_train, X_test, y_train, y_test, inciso=inciso)

    if print_results:
        for metric, value in metricas.items():
            print(f"{metric}: {value}")

def main(data: pd.DataFrame, documentos, doc_vectors,
         tokenizer, spanish_stopwords, print_results=True):

    labels = data["Polarity"].astype(int).values

    x_train_index, x_test_index, y_train, y_test = train_test_split(
        np.arange(len(doc_vectors)), labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # a)
    print("a) Sin preprocesamiento:")
    reporte_metricas(x_train_index, x_test_index, 
                     y_train, y_test, 
                     doc_vectors, print_results=print_results, inciso="a")

    # b)
    print("\nb) Con minusculas y sin stopwords:")

    documentos_minusculas = [doc.lower() for doc in documentos]
    documentos_minusculas_stp = [" ".join([token for token in doc.split() if token not in spanish_stopwords]) for doc in documentos_minusculas]
    doc_min_tokens = [tokenizer.tokenize(doc) for doc in documentos_minusculas_stp]
    w2v_min_tokens = Word2Vec(sentences=doc_min_tokens, vector_size=75, window=2, sg=1, seed=1)
    doc_min_w2v = np.array([doc2vec(doc, w2v_min_tokens) for doc in doc_min_tokens])

    reporte_metricas(x_train_index, x_test_index, 
                     y_train, y_test,
                     doc_min_w2v, print_results=print_results, inciso="b")

    # c)
    print("\nc) Con minusculas, sin stopwords y stemming:")
    stemmer = SnowballStemmer("spanish")
    doc_min_stem_tokens = [[stemmer.stem(token) for token in doc] for doc in doc_min_tokens]
    w2v_min_stem = Word2Vec(sentences=doc_min_stem_tokens, vector_size=50, window=2, sg=1, seed=1)
    doc_min_stem_w2v = np.array([doc2vec(doc, w2v_min_stem) for doc in doc_min_stem_tokens])

    reporte_metricas(x_train_index, x_test_index, 
                     y_train, y_test,
                     doc_min_stem_w2v, print_results=print_results, inciso="c")

    # d)
    print("\nd) Con minusculas, sin stopwords, stemming y frecuencia minima de 10:")
    corpus_min_stem = [token for doc_token in doc_min_stem_tokens for token in doc_token]
    fdist = FreqDist(corpus_min_stem)
    invalid_tokens = set([token for token, freq in fdist.items() if freq < 10])
    
    doc_min_stem_freq_tokens = [[token for token in doc if token not in invalid_tokens] for doc in doc_min_stem_tokens]
    w2v_min_stem_freq = Word2Vec(sentences=doc_min_stem_freq_tokens, vector_size=75, window=2, sg=1, seed=1)
    doc_min_stem_freq_w2v = np.array([doc2vec(doc, w2v_min_stem_freq) for doc in doc_min_stem_freq_tokens])

    reporte_metricas(x_train_index, x_test_index, 
                     y_train, y_test,
                     doc_min_stem_freq_w2v, print_results=print_results, inciso="d")


if __name__ == "__main__":
    from codigo.problemas.problema1 import main as p1
    from codigo.problemas.problema7 import main as p7
    from codigo.config import DATA_FILE

    tokenizer, spanish_stopwords, fdist, data, documentos, doc_tokens = p1(DATA_FILE, print_results=False)
    w2v = p7(doc_tokens, print_results=False)
    doc_vectors = np.array([doc2vec(doc, w2v) for doc in doc_tokens])
    main(data, documentos, doc_vectors, tokenizer, spanish_stopwords)
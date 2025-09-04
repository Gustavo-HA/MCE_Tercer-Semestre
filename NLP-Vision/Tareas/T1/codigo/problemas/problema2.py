import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import nltk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def obtener_logs(fdist: nltk.FreqDist):
    palabras = []
    rango = list(range(1, len(fdist) + 1))
    for palabra, _ in sorted(fdist.items(), key=lambda item: item[1], reverse=True):
        palabras.append(palabra)

    freq_palabras = [fdist[palabra] for palabra in palabras]
    
    log_freq_palabras = np.log(freq_palabras)
    log_ranking = np.log(rango)
    
    return log_freq_palabras, log_ranking

def ajuste_lineal(log_freq_palabras, log_ranking):
    X = log_ranking.reshape(-1, 1)
    y = log_freq_palabras

    modelo = LinearRegression()
    modelo.fit(X, y)

    y_pred = modelo.predict(X)
    r2 = r2_score(y, y_pred)

    return modelo, r2, y_pred

def graficar(log_freq_palabras, log_ranking, y_pred, savefig=False):
    fig, ax = plt.subplots()

    sns.lineplot(x=log_ranking, y=log_freq_palabras, ax=ax, label="Empírica")
    ax.set_xlabel("Log(Rango)")
    ax.set_ylabel("Log(Frecuencia)")
    sns.lineplot(x=log_ranking, y=y_pred.flatten(), color='red', ax=ax, label="Regresión")

    if savefig:
        plt.savefig("../figures/zipf_law_regression.pdf", bbox_inches='tight')

def main(fdist: nltk.FreqDist, savefig=False):
    log_freq_palabras, log_ranking = obtener_logs(fdist)
    modelo, r2, y_pred = ajuste_lineal(log_freq_palabras, log_ranking)
    graficar(log_freq_palabras, log_ranking, y_pred, savefig)

    print(f"s: {-modelo.coef_[0][0]}")
    print(f"C: {np.exp(modelo.intercept_[0])}")
    print(f"R^2: {r2}")

    return modelo

if __name__ == "__main__":
    from problema1 import main as p1
    from ..config import DATA_FILE

    tokenizer, spanish_stopwords, fdist, data, documentos, doc_tokens = p1(DATA_FILE, is_test=True)
    main(fdist, savefig=True)
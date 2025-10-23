import numpy as np
import matplotlib.pyplot as plt

# Inciso a)
def f(a:float,
      delta:float) -> float:
    return np.sign(a) * delta * np.floor(np.abs(a)/delta + 0.5)


# Inciso b)
def plot_cuantizada(x: np.ndarray, 
                    y: np.ndarray,
                    delta: float, 
                    figsize=(8, 2.2)):
    """
    Args:
        y: Signal to be quantized
        delta: Quantization step
        figsize: Figure size (Default value = (8, 2.2))
    """
    y_cuantizada = f(y, delta)
    plt.figure(figsize=figsize)
    plt.plot(x, y, label='Señal original')
    plt.plot(x, y_cuantizada, label='Señal cuantizada', where='mid')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid()
    plt.show()


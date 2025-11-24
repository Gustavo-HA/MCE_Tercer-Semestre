# Tarea 2: Análisis de Imágenes con Deep Learning

Este proyecto implementa y evalúa diversas arquitecturas de CNN (LeNet, AlexNet, VGG, ResNet, MobileNet, GoogLeNet) sobre los datasets MNIST y CIFAR-10.

## Requerimientos

- **uv**: Gestor de paquetes y proyectos Python.
- **GPU**: Compatible con CUDA (NVIDIA).

## Instrucciones

### Entrenamiento

Para entrenar los modelos definidos en el proyecto:

```bash
uv run src/train.py
```

### Evaluación (Test)

Para evaluar los modelos, generar matrices de confusión, calcular el número de parámetros y generar reportes de clasificación:

```bash
uv run src/test.py
```

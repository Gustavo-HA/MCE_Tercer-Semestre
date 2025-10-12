# Tarea 2: Generación y Clasificación de Texto

Este repositorio contiene la implementación de diversos modelos de aprendizaje profundo para tareas de generación y clasificación de texto. Los experimentos incluyen arquitecturas recurrentes (RNN, LSTM, GRU), convolucionales (CNN) y basadas en Transformers (Mistral-7B, mDeBERTa-v3).

## Tabla de Contenidos

- [Instalación](#instalación)
- [Parte A: Generación de Textos](#parte-a-generación-de-textos)
  - [Recolección y Preprocesamiento de Datos](#recolección-y-preprocesamiento-de-datos)
  - [Modelos RNN/LSTM/GRU](#modelos-rnnlstmgru)
  - [Modelo Mistral-7B](#modelo-mistral-7b)
- [Parte B: Clasificación de Textos](#parte-b-clasificación-de-textos)
  - [Preprocesamiento de Datos](#preprocesamiento-de-datos)
  - [Modelos RNN/LSTM/GRU](#modelos-rnnlstmgru-1)
  - [Modelo CNN](#modelo-cnn)
  - [Modelo mDeBERTa-v3](#modelo-mdeberta-v3)

---

## Instalación

Este proyecto utiliza `uv` para la gestión de dependencias mediante `pyproject.toml`. Como alternativa, también puede utilizarse `venv` con el archivo `requirements.txt`.

### Opción 1: Usando `uv` (Recomendado)

```bash
# Instalar uv si no lo tienes
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sincronizar dependencias desde pyproject.toml
uv sync
```

### Opción 2: Usando `venv` con `requirements.txt`

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
source venv/bin/activate  # En Linux/macOS
# venv\Scripts\activate   # En Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## Parte A: Generación de Textos

### Recolección y Preprocesamiento de Datos

#### 1. Scraping de Letras de Canciones

El script `scrape_songs_alpaca.py` recolecta letras de canciones desde Genius.com utilizando su API y las formatea en formato Alpaca para fine-tuning de modelos.

```bash
python codigo/generative/scraping/scrape_songs_alpaca.py
```

**Nota:** Requiere un token de API de Genius configurado en un archivo `.env` con la variable `GENIUS_API_TOKEN`.

#### 2. Preprocesamiento de Datos

El script `preprocess_scraped_songs.py` procesa las letras recolectadas y genera datasets para entrenamiento a nivel de caracteres y palabras.

```bash
python codigo/generative/scraping/preprocess_scraped_songs.py
```

Este script genera:
- `train_lyrics.txt` y `test_lyrics.txt`: Datos procesados para modelos a nivel de carácter/palabra
- `train_lyrics_alpaca.json` y `test_lyrics_alpaca.json`: Datos en formato Alpaca para Mistral
- `vocab_char.json` y `vocab_word.json`: Vocabularios para RNN/LSTM/GRU

---

### Modelos RNN/LSTM/GRU

#### Entrenamiento

El script `training.py` entrena modelos recurrentes para generación de texto a nivel de caracteres o palabras.

```bash
# Ejemplo: LSTM a nivel de caracter.
python codigo/generative/rnn-lstm-gru/training.py \
    --rnn_type LSTM \
    --level char \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --num_layers 2 \
    --dropout 0.3 \
    --batch_size 64 \
    --seq_length 100 \
    --learning_rate 0.002 \
    --num_epochs 20
```

**Parámetros principales:**
- `--rnn_type`: Tipo de RNN (`RNN`, `LSTM`, `GRU`) - **Default:** `LSTM`
- `--level`: Nivel de tokenización (`char`, `word`) - **Default:** `char`
- `--embedding_dim`: Dimensión de embeddings - **Default:** `128`
- `--hidden_dim`: Dimensión del estado oculto - **Default:** `256`
- `--num_layers`: Número de capas recurrentes - **Default:** `2`
- `--dropout`: Probabilidad de dropout - **Default:** `0.3`
- `--batch_size`: Tamaño de lote - **Default:** `64`
- `--seq_length`: Longitud de secuencia - **Default:** `100`
- `--learning_rate`: Tasa de aprendizaje - **Default:** `0.002`
- `--num_epochs`: Número de épocas - **Default:** `20`

Los valores predeterminados corresponden a la configuración utilizada en el reporte.

#### Inferencia

El script `inference.py` genera texto nuevo usando un modelo entrenado.

```bash
python codigo/generative/rnn-lstm-gru/inference.py \
    --model_path models/text-gen/LSTM/checkpoint_epoch_20.pt \
    --level char \
    --rnn_type LSTM \
    --length 500 \
    --temperature 0.8
```

**Parámetros principales:**
- `--model_path`: Ruta al checkpoint del modelo (requerido)
- `--level`: Nivel de tokenización (`char`, `word`) - requerido
- `--rnn_type`: Tipo de RNN - **Default:** `LSTM`
- `--length`: Número de tokens a generar - **Default:** `500`
- `--temperature`: Temperatura de muestreo - **Default:** `0.8`
- `--vocab_path`: Ruta al vocabulario (auto-detecta si no se especifica)
- `--output_file`: Archivo de salida (imprime en consola si no se especifica)

#### Cálculo de Perplejidad

El script `perplexity.py` calcula la perplejidad del modelo en el conjunto de prueba.

```bash
python codigo/generative/rnn-lstm-gru/perplexity.py \
    --model_path models/text-gen/LSTM/checkpoint_epoch_20.pt \
    --level char \
    --block_size 100
```

**Parámetros principales:**
- `--model_path`: Ruta al checkpoint del modelo (requerido)
- `--level`: Nivel de tokenización (`char`, `word`) - requerido
- `--test_file`: Archivo de prueba (auto-detecta si no se especifica)
- `--vocab_path`: Ruta al vocabulario (auto-detecta si no se especifica)
- `--block_size`: Tamaño de bloque para streaming - **Default:** `100`

---

### Modelo Mistral-7B

#### Fine-tuning

El script `fine-tune_mistral.py` realiza fine-tuning de Mistral-7B-Instruct usando LoRA y Unsloth.

```bash
python codigo/generative/mistral/fine-tune_mistral.py \
    --lr 2e-4 \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 8 \
    --lora_r 32 \
    --max_seq_length 2048
```

**Parámetros principales:**
- `--lr`: Tasa de aprendizaje - **Default:** `2e-4`
- `--epochs`: Número de épocas - **Default:** `3`
- `--batch_size`: Tamaño de lote por dispositivo - **Default:** `2`
- `--grad_accum`: Pasos de acumulación de gradiente - **Default:** `8`
- `--lora_r`: Rango de LoRA - **Default:** `32`
- `--max_seq_length`: Longitud máxima de secuencia - **Default:** `2048`
- `--output_dir`: Nombre del directorio de salida (auto-generado si no se especifica)

Los valores predeterminados corresponden a la configuración utilizada en el reporte.

#### Inferencia

El script `inference_mistral.py` genera letras de canciones usando el modelo fine-tuned.

```bash
python codigo/generative/mistral/inference_mistral.py \
    --model_dir models/text-gen/mistral_lr0.0002_ep3_bs2x8_r32_20251003_124706/final_model \
    --artist "Kendrick Lamar" \
    --max_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --repetition_penalty 1.1
```

**Parámetros principales:**
- `--model_dir`: Ruta al modelo fine-tuned (requerido)
- `--artist`: Estilo del artista - **Default:** `"Kendrick Lamar"`
- `--prompt`: Prompt personalizado (opcional)
- `--max_tokens`: Tokens máximos a generar - **Default:** `512`
- `--temperature`: Temperatura de muestreo - **Default:** `0.7`
- `--top_p`: Top-p sampling - **Default:** `0.9`
- `--repetition_penalty`: Penalización por repetición - **Default:** `1.1`

#### Cálculo de Perplejidad

El script `calculate_perplexity.py` calcula la perplejidad del modelo fine-tuned.

```bash
python codigo/generative/mistral/calculate_perplexity.py \
    --model_dir models/text-gen/mistral_lr0.0002_ep3_bs2x8_r32_20251003_124706/final_model \
    --test_file data/text_gen/test_lyrics_alpaca.json
```

**Parámetros principales:**
- `--model_dir`: Ruta al modelo fine-tuned (requerido)
- `--test_file`: Archivo de prueba en formato Alpaca (requerido)
- `--batch_size`: Tamaño de lote - **Default:** `4`
- `--max_length`: Longitud máxima de secuencia - **Default:** `2048`

---

## Parte B: Clasificación de Textos

### Preprocesamiento de Datos

El script `prepare_data.py` preprocesa el dataset REST-MEX 2025 para clasificación de sentimientos (1-5 estrellas).

```bash
python codigo/classification/prepare_data.py \
    --file_path data/classification/meia_data.csv \
    --output_path data/classification/meia_data.json
```

**Parámetros principales:**
- `--file_path`: Ruta al archivo CSV de entrada (requerido)
- `--output_path`: Ruta del archivo JSON de salida (requerido)
- `--min_freq`: Frecuencia mínima para incluir palabra en vocabulario - **Default:** `2`
- `--max_vocab_size`: Tamaño máximo del vocabulario (opcional)

Este script genera:
- `meia_data.json`: Datos en formato JSON para mDeBERTa
- `meia_data_train.csv`, `meia_data_val.csv`, `meia_data_test.csv`: Datos CSV para RNN/LSTM/GRU/CNN
- `meia_data_vocab.json`: Vocabulario para modelos from-scratch
- `meia_data_stats.json`: Estadísticas de longitud de secuencias

---

### Modelos RNN/LSTM/GRU

#### Entrenamiento

El script `train.py` entrena modelos recurrentes para clasificación de sentimientos.

```bash
python codigo/classification/rnn-lstm-gru/train.py \
    --rnn_type LSTM \
    --lr 1e-3 \
    --epochs 50 \
    --batch_size 32 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --num_layers 2 \
    --dropout 0.3 \
    --max_length 128 \
    --patience 15
```

**Parámetros principales:**
- `--rnn_type`: Tipo de RNN (`RNN`, `LSTM`, `GRU`) - **Default:** `LSTM`
- `--lr`: Tasa de aprendizaje - **Default:** `1e-3`
- `--epochs`: Número de épocas - **Default:** `50`
- `--batch_size`: Tamaño de lote - **Default:** `32`
- `--embedding_dim`: Dimensión de embeddings - **Default:** `128`
- `--hidden_dim`: Dimensión del estado oculto - **Default:** `256`
- `--num_layers`: Número de capas - **Default:** `2`
- `--dropout`: Probabilidad de dropout - **Default:** `0.3`
- `--max_length`: Longitud máxima de secuencia - **Default:** `128`
- `--patience`: Paciencia para early stopping - **Default:** `15`
- `--vocab_path`: Ruta al vocabulario - **Default:** `./data/classification/meia_data_vocab.json`
- `--train_path`: Ruta a datos de entrenamiento - **Default:** `./data/classification/meia_data_train.csv`
- `--val_path`: Ruta a datos de validación - **Default:** `./data/classification/meia_data_val.csv`
- `--test_path`: Ruta a datos de prueba - **Default:** `./data/classification/meia_data_test.csv`

Los valores predeterminados corresponden a la configuración utilizada en el reporte.

**Nota:** Estos modelos no tienen script de inferencia separado; la evaluación se realiza automáticamente al final del entrenamiento.

---

### Modelo CNN

#### Entrenamiento

El script `train.py` entrena una CNN (TextCNN) para clasificación de sentimientos.

```bash
python codigo/classification/cnn/train.py \
    --lr 1e-3 \
    --epochs 50 \
    --batch_size 128 \
    --embedding_dim 128 \
    --num_filters 100 \
    --kernel_sizes 3 4 5 \
    --dropout 0.5 \
    --max_length 128 \
    --patience 10
```

**Parámetros principales:**
- `--lr`: Tasa de aprendizaje - **Default:** `1e-3`
- `--epochs`: Número de épocas - **Default:** `50`
- `--batch_size`: Tamaño de lote - **Default:** `128`
- `--embedding_dim`: Dimensión de embeddings - **Default:** `128`
- `--num_filters`: Número de filtros por kernel - **Default:** `100`
- `--kernel_sizes`: Tamaños de kernels - **Default:** `[3, 4, 5]`
- `--dropout`: Probabilidad de dropout - **Default:** `0.5`
- `--max_length`: Longitud máxima de secuencia - **Default:** `128`
- `--patience`: Paciencia para early stopping - **Default:** `10`
- `--vocab_path`: Ruta al vocabulario - **Default:** `./data/classification/meia_data_vocab.json`
- `--train_path`: Ruta a datos de entrenamiento - **Default:** `./data/classification/meia_data_train.csv`
- `--val_path`: Ruta a datos de validación - **Default:** `./data/classification/meia_data_val.csv`
- `--test_path`: Ruta a datos de prueba - **Default:** `./data/classification/meia_data_test.csv`

Los valores predeterminados corresponden a la configuración utilizada en el reporte.

**Nota:** Este modelo no tiene script de inferencia separado; la evaluación se realiza automáticamente al final del entrenamiento.

---

### Modelo mDeBERTa-v3

#### Fine-tuning

El script `train.py` realiza fine-tuning de mDeBERTa-v3-base para clasificación de sentimientos.

```bash
python codigo/classification/transformer/train.py \
    --lr 2e-5 \
    --epochs 3 \
    --batch_size 8 \
    --grad_accum 4 \
    --max_seq_length 512 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --save_steps 100 \
    --eval_steps 100
```

**Parámetros principales:**
- `--lr`: Tasa de aprendizaje - **Default:** `2e-5`
- `--epochs`: Número de épocas - **Default:** `3`
- `--batch_size`: Tamaño de lote por dispositivo - **Default:** `8`
- `--grad_accum`: Pasos de acumulación de gradiente - **Default:** `4`
- `--max_seq_length`: Longitud máxima de secuencia - **Default:** `512`
- `--warmup_ratio`: Proporción de warmup - **Default:** `0.1`
- `--weight_decay`: Decaimiento de pesos - **Default:** `0.01`
- `--save_steps`: Guardar checkpoint cada N pasos - **Default:** `100`
- `--eval_steps`: Evaluar cada N pasos - **Default:** `100`
- `--seed`: Semilla aleatoria - **Default:** `42`

Los valores predeterminados corresponden a la configuración utilizada en el reporte.

**Nota:** El modelo carga datos desde `./data/classification/meia_data.json`. Este modelo no tiene script de inferencia separado; la evaluación se realiza automáticamente al final del entrenamiento.

---

## Estructura del Proyecto

```
.
├── codigo/
│   ├── classification/
│   │   ├── prepare_data.py           # Preprocesamiento para clasificación
│   │   ├── rnn-lstm-gru/
│   │   │   └── train.py              # Entrenamiento RNN/LSTM/GRU
│   │   ├── cnn/
│   │   │   └── train.py              # Entrenamiento CNN
│   │   └── transformer/
│   │       └── train.py              # Fine-tuning mDeBERTa-v3
│   └── generative/
│       ├── scraping/
│       │   ├── scrape_songs_alpaca.py      # Scraping de letras
│       │   └── preprocess_scraped_songs.py # Preprocesamiento
│       ├── rnn-lstm-gru/
│       │   ├── training.py           # Entrenamiento RNN/LSTM/GRU
│       │   ├── inference.py          # Generación de texto
│       │   └── perplexity.py         # Cálculo de perplejidad
│       └── mistral/
│           ├── fine-tune_mistral.py  # Fine-tuning Mistral
│           ├── inference_mistral.py  # Generación con Mistral
│           └── calculate_perplexity.py # Perplejidad Mistral
├── data/
│   ├── classification/               # Datos de clasificación
│   └── text_gen/                     # Datos de generación
├── models/                           # Checkpoints de modelos
├── pyproject.toml                    # Configuración de dependencias
├── requirements.txt                  # Dependencias para venv
└── README.md                         # Este archivo
```

---

## Notas Adicionales

- **Hardware:** Los experimentos se ejecutaron en 2× NVIDIA RTX TITAN (24 GB VRAM cada una), 128 GB RAM y 24 núcleos CPU Intel Xeon Silver 4214 @ 2.2 GHz.
- **Reproducibilidad:** Todos los scripts utilizan semillas aleatorias (default: 42) para garantizar reproducibilidad.
- **Checkpoints:** Los modelos se guardan automáticamente durante el entrenamiento con early stopping habilitado.
- **Logs:** Los scripts generan logs detallados de métricas, uso de memoria y progreso del entrenamiento.

---

## Referencias

Para más detalles sobre la implementación, arquitecturas y resultados experimentales, consultar el documento técnico en `documento/reporte.pdf`.

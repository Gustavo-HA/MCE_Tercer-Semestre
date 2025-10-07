#!/bin/bash

# Partición en la que se va a ejecutar (cambia a C1 o C2 para CPU)
#SBATCH --partition=GPU

# Nombre del trabajo
#SBATCH --job-name=RNN-LSTM-GRU_Perplexity

# Número de tareas
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Archivo de log donde quedará lo que imprima su software por pantalla
#SBATCH --output="/home/est_posgrado_gustavo.angeles/Tercer Semestre/NLP-Vision/Tareas/T2/logs/perplexity.log"

# Archivo de error donde se guardarán los errores del trabajo
#SBATCH --error="/home/est_posgrado_gustavo.angeles/Tercer Semestre/NLP-Vision/Tareas/T2/logs/perplexity.err"

# Memoria máxima a utilizar
#SBATCH --mem=0

# Tiempo máximo de ejecución
#SBATCH --time=02:00:00

# Enviar correo electrónico cuando el trabajo finalice o falle
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gustavo.angeles@cimat.mx


######################### INICIO DEL SCRIPT DE SLURM #######################

echo "--- Inicio del trabajo de Slurm ---"
echo "Ejecutándose en el host: $(hostname)"

####################### CONFIGURACION DE UV #######################

# add uv to path
export PATH="/home/est_posgrado_gustavo.angeles/.local/uv/bin:$PATH"

# change to the directory where the job was submitted
cd "/home/est_posgrado_gustavo.angeles/Tercer Semestre/NLP-Vision/Tareas/T2"

uv sync
source ./.venv/bin/activate

if [ $? -ne 0 ]; then
    echo "ERROR: No se pudo activar el entorno virtual."
    exit 1
fi

echo "Entorno virtual activado correctamente."

####################### CALCULO DE PERPLEJIDAD #######################

# Define model paths - Change these to your actual model paths
CHAR_MODEL="models/RNN/char/best_model.pt"
WORD_MODEL="models/RNN/word/best_model.pt"

echo "=========================================="
echo " Simple RNN"
echo "=========================================="

echo ""
echo "MODELO CHAR"
echo ""

# Calculate perplexity for character-level model
python codigo/generative/rnn-lstm-gru/perplexity.py \
    --model_path "$CHAR_MODEL" \
    --level char 

echo ""
echo "MODELO WORD"
echo ""

# Calculate perplexity for word-level model
python codigo/generative/rnn-lstm-gru/perplexity.py \
    --model_path "$WORD_MODEL" \
    --level word 

echo ""
echo "=========================================="
echo " RNN LSTM "
echo "=========================================="
echo ""

CHAR_MODEL="models/LSTM/char/best_model.pt"
WORD_MODEL="models/LSTM/word/best_model.pt"

echo ""
echo "MODELO CHAR"
echo ""

# Calculate perplexity for character-level model
python codigo/generative/rnn-lstm-gru/perplexity.py \
    --model_path "$CHAR_MODEL" \
    --level char

echo ""
echo "MODELO WORD"
echo ""

# Calculate perplexity for word-level model
python codigo/generative/rnn-lstm-gru/perplexity.py \
    --model_path "$WORD_MODEL" \
    --level word 


echo ""
echo "=========================================="
echo " RNN GRU "
echo "=========================================="
echo ""

CHAR_MODEL="models/GRU/char/best_model.pt"
WORD_MODEL="models/GRU/word/best_model.pt"

echo ""
echo "MODELO CHAR"
echo ""

# Calculate perplexity for character-level model
python codigo/generative/rnn-lstm-gru/perplexity.py \
    --model_path "$CHAR_MODEL" \
    --level char 

echo ""
echo "MODELO WORD"
echo ""

# Calculate perplexity for word-level model
python codigo/generative/rnn-lstm-gru/perplexity.py \
    --model_path "$WORD_MODEL" \
    --level word


echo "--- Fin del trabajo de Slurm ---"

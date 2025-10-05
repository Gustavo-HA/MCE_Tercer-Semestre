#!/bin/bash

# Partición en la que se va a ejecutar (cambia a C1 o C2 para CPU)
#SBATCH --partition=GPU

# Nombre del trabajo
#SBATCH --job-name=RNN_Perplexity

# Número de tareas
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Archivo de log donde quedará lo que imprima su software por pantalla
#SBATCH --output="/home/est_posgrado_gustavo.angeles/Tercer Semestre/NLP-Vision/Tareas/T2/logs/rnn_perplexity.log"

# Archivo de error donde se guardarán los errores del trabajo
#SBATCH --error="/home/est_posgrado_gustavo.angeles/Tercer Semestre/NLP-Vision/Tareas/T2/logs/rnn_perplexity.err"

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
CHAR_MODEL="models/rnn/char/best_model.pt"
WORD_MODEL="models/rnn/word/best_model.pt"

echo ""
echo "=========================================="
echo "CALCULANDO PERPLEJIDAD - MODELO CHAR"
echo "=========================================="
echo ""

# Calculate perplexity for character-level model
python codigo/generative/rnn/rnn_perplexity.py \
    --model_path "$CHAR_MODEL" \
    --level char \
    --batch_size 64 \
    --seq_length 256 \
    --method dataloader

echo ""
echo "=========================================="
echo "CALCULANDO PERPLEJIDAD - MODELO WORD"
echo "=========================================="
echo ""

# Calculate perplexity for word-level model
python codigo/generative/rnn/rnn_perplexity.py \
    --model_path "$WORD_MODEL" \
    --level word \
    --batch_size 64 \
    --seq_length 256 \
    --method dataloader

echo ""
echo "=========================================="
echo "CÁLCULO DE PERPLEJIDAD COMPLETADO"
echo "=========================================="
echo ""

echo "--- Fin del trabajo de Slurm ---"

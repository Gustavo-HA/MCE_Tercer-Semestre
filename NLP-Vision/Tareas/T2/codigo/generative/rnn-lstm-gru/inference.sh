#!/bin/bash

# Partición en la que se va a ejecutar (cambia a C1 o C2 para CPU)
#SBATCH --partition=GPU

# Nombre del trabajo
#SBATCH --job-name=RNN-LSTM-GRU_Inference

# Número de tareas
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Archivo de log donde quedará lo que imprima su software por pantalla
#SBATCH --output="/home/est_posgrado_gustavo.angeles/Tercer Semestre/NLP-Vision/Tareas/T2/logs/inference.log"

# Archivo de error donde se guardarán los errores del trabajo
#SBATCH --error="/home/est_posgrado_gustavo.angeles/Tercer Semestre/NLP-Vision/Tareas/T2/logs/inference.err"

# Memoria máxima a utilizar
#SBATCH --mem=0

# Tiempo máximo de ejecución
#SBATCH --time=10:00:00

# Enviar correo electrónico cuando el trabajo finalice o falle
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gustavo.angeles@cimat.mx


######################### INICIO DEL SCRIPT DE SLURM #######################

echo "--- Inicio del trabajo de Slurm ---"
echo "Ejecutándose en el host: $(hostname)"

####################### CONFIGURACION DE UV #######################

# add uv to path
#export PATH="/home/est_posgrado_gustavo.angeles/.local/uv/bin:$PATH"

# change to the directory where the job was submitted
#cd "/home/est_posgrado_gustavo.angeles/Tercer Semestre/NLP-Vision/Tareas/T2"

uv sync
source ./.venv/bin/activate

if [ $? -ne 0 ]; then
    echo "ERROR: No se pudo activar el entorno virtual."
    exit 1
fi

####################### EJECUCION DEL SCRIPT DE PYTHON #######################

PYTHON_SCRIPT="./codigo/generative/rnn-lstm-gru/inference.py"

echo "Ejecutando el script de Python: $PYTHON_SCRIPT"

# A nivel de Palabra ############################
NIVEL="word"
RNN_TYPE="RNN"
python "$PYTHON_SCRIPT" --model_path "./models/${RNN_TYPE}/${NIVEL}/best_model.pt" \
    --level "$NIVEL" --rnn_type "$RNN_TYPE" --length 500 --temperature 0.3 \
    --output_file "./data/text_gen/inferences/${RNN_TYPE}_${NIVEL}_inference.txt"

RNN_TYPE="LSTM"
python "$PYTHON_SCRIPT" --model_path "./models/${RNN_TYPE}/${NIVEL}/best_model.pt" \
    --level "$NIVEL" --rnn_type "$RNN_TYPE" --length 500 --temperature 0.3 \
    --output_file "./data/text_gen/inferences/${RNN_TYPE}_${NIVEL}_inference.txt"

RNN_TYPE="GRU"
python "$PYTHON_SCRIPT" --model_path "./models/${RNN_TYPE}/${NIVEL}/best_model.pt" \
    --level "$NIVEL" --rnn_type "$RNN_TYPE" --length 500 --temperature 0.3 \
    --output_file "./data/text_gen/inferences/${RNN_TYPE}_${NIVEL}_inference.txt"

# A nivel de Caracter ############################
NIVEL="char"

RNN_TYPE="RNN"
python "$PYTHON_SCRIPT" --model_path "./models/${RNN_TYPE}/${NIVEL}/best_model.pt" \
    --level "$NIVEL" --rnn_type "$RNN_TYPE" --length 500 --temperature 0.3 \
    --output_file "./data/text_gen/inferences/${RNN_TYPE}_${NIVEL}_inference.txt"

RNN_TYPE="LSTM"
python "$PYTHON_SCRIPT" --model_path "./models/${RNN_TYPE}/${NIVEL}/best_model.pt" \
    --level "$NIVEL" --rnn_type "$RNN_TYPE" --length 500 --temperature 0.3 \
    --output_file "./data/text_gen/inferences/${RNN_TYPE}_${NIVEL}_inference.txt"

RNN_TYPE="GRU"
python "$PYTHON_SCRIPT" --model_path "./models/${RNN_TYPE}/${NIVEL}/best_model.pt" \
    --level "$NIVEL" --rnn_type "$RNN_TYPE" --length 500 --temperature 0.3 \
    --output_file "./data/text_gen/inferences/${RNN_TYPE}_${NIVEL}_inference.txt"


if [ $? -ne 0 ]; then
    echo "ERROR: El script de Python falló."
    exit 1
fi

echo "--- Trabajo de Slurm Finalizado ---"
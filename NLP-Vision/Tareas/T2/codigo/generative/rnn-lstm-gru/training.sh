#!/bin/bash

# Partición en la que se va a ejecutar (cambia a C1 o C2 para CPU)
#SBATCH --partition=GPU

# Nombre del trabajo
#SBATCH --job-name=RNN-LSTM-GRU_Training

# Número de tareas
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Archivo de log donde quedará lo que imprima su software por pantalla
#SBATCH --output="/home/est_posgrado_gustavo.angeles/Tercer Semestre/NLP-Vision/Tareas/T2/logs/training.log"

# Archivo de error donde se guardarán los errores del trabajo
#SBATCH --error="/home/est_posgrado_gustavo.angeles/Tercer Semestre/NLP-Vision/Tareas/T2/logs/training.err"

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
export PATH="/home/est_posgrado_gustavo.angeles/.local/uv/bin:$PATH"

# change to the directory where the job was submitted
cd "/home/est_posgrado_gustavo.angeles/Tercer Semestre/NLP-Vision/Tareas/T2"

uv sync
source ./.venv/bin/activate

if [ $? -ne 0 ]; then
    echo "ERROR: No se pudo activar el entorno virtual."
    exit 1
fi

####################### EJECUCION DEL SCRIPT DE PYTHON #######################

PYTHON_SCRIPT="./codigo/generative/rnn-lstm-gru/training.py"

echo "Ejecutando el script de Python: $PYTHON_SCRIPT"

python "$PYTHON_SCRIPT" --level word --num_epochs 20 --rnn_type LSTM # Uses default hyperparameters
python "$PYTHON_SCRIPT" --level word --num_epochs 20 --rnn_type GRU # Uses default hyperparameters
python "$PYTHON_SCRIPT" --level word --num_epochs 20 --rnn_type RNN # Uses default hyperparameters

python "$PYTHON_SCRIPT" --level char --num_epochs 20 --rnn_type LSTM # Uses default hyperparameters
python "$PYTHON_SCRIPT" --level char --num_epochs 20 --rnn_type GRU # Uses default hyperparameters
python "$PYTHON_SCRIPT" --level char --num_epochs 20 --rnn_type RNN # Uses default hyperparameters

if [ $? -ne 0 ]; then
    echo "ERROR: El script de Python falló."
    exit 1
fi

echo "--- Trabajo de Slurm Finalizado ---"
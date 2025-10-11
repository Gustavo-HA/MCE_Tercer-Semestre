#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --job-name=CNN_training_class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="/home/est_posgrado_gustavo.angeles/Tercer Semestre/NLP-Vision/Tareas/T2/logs/training-cnn_class.log"
#SBATCH --error="/home/est_posgrado_gustavo.angeles/Tercer Semestre/NLP-Vision/Tareas/T2/logs/training-cnn_class.err"
#SBATCH --mem=0
#SBATCH --time=10:00:00
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

PYTHON_SCRIPT="./codigo/classification/cnn/train.py"

echo "Ejecutando el script de Python: $PYTHON_SCRIPT"

python "$PYTHON_SCRIPT" 

if [ $? -ne 0 ]; then
    echo "ERROR: El script de Python falló."
    exit 1
fi

echo "--- Trabajo de Slurm Finalizado ---"
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --job-name=CNNs_Training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=0

#SBATCH --output=logs/training-cnns.log
#SBATCH --error=logs/training-cnns.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gustavo.angeles@cimat.mx

set -euo pipefail

# --- Configuracion de entorno ---
export PATH="/home/est_posgrado_gustavo.angeles/.local/uv/bin:$PATH"
cd "/home/est_posgrado_gustavo.angeles/Tercer Semestre/NLP-Vision/Tareas/visionT2"

# Asegurar dependencias
uv sync

# Crear carpeta de logs si no existe
mkdir -p logs

# --- Iniciar MLflow local ---
MLRUNS_DIR="$(pwd)/mlruns"
mkdir -p "$MLRUNS_DIR"

uv run mlflow server \
	--backend-store-uri "file:$MLRUNS_DIR" \
	--default-artifact-root "file:$MLRUNS_DIR" \
	--host 127.0.0.1 \
	--port 1825 \
	--serve-artifacts \
	>/dev/null 2>&1 &
MLFLOW_PID=$!
echo "[INFO] MLflow iniciado en http://127.0.0.1:1825 (PID=$MLFLOW_PID)"
sleep 5

# --- Ejecutar entrenamiento ---
uv run python ./src/train.py

# --- Detener MLflow ---
if ps -p "$MLFLOW_PID" >/dev/null 2>&1; then
	kill "$MLFLOW_PID" || true
fi
echo "[INFO] Entrenamiento finalizado."
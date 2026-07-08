#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "==================================================="
echo "  MoneyPrinterTurbo Interface Auto-Launcher (macOS)"
echo "==================================================="

# Get the directory where the script is located
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

echo "[1/3] Checking backend virtual environment..."
if [ ! -d "$BACKEND_DIR/.venv" ]; then
    echo "Virtual environment not found. Creating one..."
    if command -v uv &> /dev/null; then
        echo "Using uv to create virtual environment with Python 3.10..."
        uv venv --python 3.10 "$BACKEND_DIR/.venv"
    else
        echo "Using python3 to create virtual environment..."
        python3 -m venv "$BACKEND_DIR/.venv"
    fi
fi

echo "[2/3] Installing/upgrading backend dependencies..."
source "$BACKEND_DIR/.venv/bin/activate"

if command -v uv &> /dev/null; then
    echo "Installing/verifying backend dependencies using uv..."
    uv pip install --python "$BACKEND_DIR/.venv" -r "$BACKEND_DIR/requirements.txt"
    # Install numpy first — required as build dependency for pkuseg (via chatterbox-tts)
    uv pip install --python "$BACKEND_DIR/.venv" numpy
    uv pip install --python "$BACKEND_DIR/.venv" psycopg2-binary Pillow moviepy edge-tts torch torchaudio transformers diffusers
    uv pip install --python "$BACKEND_DIR/.venv" chatterbox-tts --no-build-isolation
else
    echo "Installing/verifying backend dependencies using pip..."
    python3 -m pip install --upgrade pip
    pip install -r "$BACKEND_DIR/requirements.txt"
    pip install numpy
    pip install psycopg2-binary Pillow moviepy edge-tts torch torchaudio transformers diffusers
    pip install chatterbox-tts --no-build-isolation
fi

echo "[3/3] Checking frontend dependencies..."
cd "$FRONTEND_DIR"
if [ ! -d "node_modules" ]; then
    echo "Frontend node_modules not found. Running npm install..."
    npm install
else
    echo "Frontend node_modules already exists. Skipping npm install."
fi

# Function to clean up background processes on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill "$BACKEND_PID" 2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

echo "==================================================="
echo "  Starting Backend and Frontend concurrently..."
echo "==================================================="

# Start FastAPI Backend in background
echo "Starting FastAPI Backend on http://localhost:8000 ..."
cd "$BACKEND_DIR"
python3 main.py &
BACKEND_PID=$!

# Start Vite Frontend
echo "Starting Vite Frontend on http://localhost:5173 ..."
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!

echo "==================================================="
echo "  Both servers have been launched!"
echo "  - Backend: http://localhost:8000"
echo "  - Frontend: http://localhost:5173"
echo "  Press Ctrl+C to stop both servers."
echo "==================================================="

# Keep the script running to monitor children
wait

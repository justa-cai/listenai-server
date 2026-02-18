#!/bin/bash
# Start script for VoxCPM TTS Server

set -e

rm -rf ./tmp/*

# Default values
HOST="${TTS_HOST:-0.0.0.0}"
PORT="${TTS_PORT:-9300}"
LOG_LEVEL="${TTS_LOG_LEVEL:-INFO}"
LOG_FORMAT="${TTS_LOG_FORMAT:-json}"

# Set environment variables for PyTorch compatibility
# PyTorch 2.6+ requires weights_only=False for legacy .tar format models
export TORCH_WEIGHTS_ONLY=0

# Disable torch.compile to avoid TLS issues in async executor
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment from .env file"
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "Starting VoxCPM TTS Server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Log Level: $LOG_LEVEL"
echo "Log Format: $LOG_FORMAT"

# Run the server
python -m src.main

#!/bin/bash
# Stop script for VoxCPM TTS Server

echo "Stopping VoxCPM TTS Server..."

# Find and kill the server process
PID=$(pgrep -f "python -m src.main" || true)

if [ -n "$PID" ]; then
    echo "Killing server process (PID: $PID)"
    kill $PID
    echo "Server stopped"
else
    echo "Server is not running"
fi

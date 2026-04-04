#!/bin/bash

# ANIMA-bot RAG System Runner

# 1. Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# 2. Activate virtual environment
source venv/bin/activate

# 3. Load config to check backend
CONFIG_FILE="${ANIMA_CONFIG:-./config.yaml}"
if [ -f "$CONFIG_FILE" ]; then
    BACKEND=$(grep -E "^\s*backend:" "$CONFIG_FILE" | head -1 | awk '{print $2}' | tr -d '"')
else
    BACKEND="ollama"
fi

# 4. Check if Ollama server is running (only if using ollama backend)
if [ "$BACKEND" = "ollama" ]; then
    echo "Checking Ollama server..."
    if ! curl -s --connect-timeout 5 http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Warning: Ollama server is not responding at localhost:11434"
        echo "Attempting to start Ollama..."

        # Try to start Ollama in the background
        if command -v ollama &> /dev/null; then
            ollama serve &
            sleep 3

            # Verify it started
            if ! curl -s --connect-timeout 5 http://localhost:11434/api/tags > /dev/null 2>&1; then
                echo "Error: Could not start Ollama server."
                echo "Please start Ollama manually: 'ollama serve' or open the Ollama app."
                exit 1
            fi
            echo "Ollama server started successfully."
        else
            echo "Error: Ollama command not found."
            echo "Please install Ollama from https://ollama.com/ or switch to a different backend in config.yaml"
            exit 1
        fi
    else
        echo "Ollama server is running."
    fi
fi

# 5. Run the system
echo "Starting ANIMA-bot RAG System..."
python3 app.py

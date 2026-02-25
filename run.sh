#!/bin/bash

# ANIMA-bot RAG System Runner

# 1. Check if Ollama server is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Error: Ollama server is not running."
    echo "Please start the Ollama application or run 'ollama serve' in another terminal."
    exit 1
fi

# 2. Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# 3. Activate virtual environment and run the system
echo "Starting ANIMA-bot RAG System..."
source venv/bin/activate
python3 rag_system.py

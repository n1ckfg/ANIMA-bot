#!/bin/bash

# ANIMA-bot RAG System Runner

# 1. Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# 2. Activate virtual environment and run the system
echo "Starting ANIMA-bot RAG System..."
source venv/bin/activate
python3 rag_system.py

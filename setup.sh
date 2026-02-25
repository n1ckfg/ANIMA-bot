#!/bin/bash

# ANIMA-bot RAG System Setup Script

# 1. Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo "Python 3 could not be found. Please install it first."
    exit 1
fi

# 2. Check for Ollama
if ! command -v ollama &> /dev/null
then
    echo "Ollama could not be found. Please install it from https://ollama.com/"
    exit 1
fi

# 3. Pull required LLM model (llama3.1:8b)
echo "Ensuring llama3.1:8b is available in Ollama..."
ollama pull llama3.1:8b

# 4. Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# 5. Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 6. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 7. Install dependencies
echo "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Please ensure it exists in the current directory."
    exit 1
fi

# 8. Download and save the embedding model locally
echo "Downloading and saving the local embedding model (bge-small-en-v1.5)..."
if [ ! -d "models/bge-small-en-v1.5" ]; then
    mkdir -p models
    python3 -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-small-en-v1.5'); model.save('models/bge-small-en-v1.5')"
fi

# 9. Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir data
fi

echo "Setup complete!"
echo "To start the RAG system, run: source venv/bin/activate && python3 rag_system.py"

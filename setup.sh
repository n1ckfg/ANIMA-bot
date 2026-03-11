#!/bin/bash

# ANIMA-bot RAG System Setup Script

#CHAT_MODEL="qwen3:4b-instruct"
CHAT_MODEL="llama3.1:8b"

DATA_MODEL_PROVIDER="BAAI"
#DATA_MODEL="bge-small-en-v1.5"
DATA_MODEL="bge-m3"

RERANKER_MODEL_PROVIDER="BAAI"
RERANKER_MODEL="bge-reranker-v2-m3"

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

# 3. Pull required LLM model
echo "Ensuring $CHAT_MODEL is available in Ollama..."
ollama pull $CHAT_MODEL

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

# 8a. Download and save the embedding model
echo "Downloading and saving the embedding model ($DATA_MODEL)..."
if [ ! -d "models/$DATA_MODEL" ]; then
    mkdir -p models
    python3 -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('$DATA_MODEL_PROVIDER/$DATA_MODEL'); model.save('models/$DATA_MODEL')"
fi

# 8b. Download and save the reranker model
echo "Downloading and saving the reranker model ($RERANKER_MODEL)..."
if [ ! -d "models/$RERANKER_MODEL" ]; then
    mkdir -p models
    python3 -c "from sentence_transformers import CrossEncoder; model = CrossEncoder('$RERANKER_MODEL_PROVIDER/$RERANKER_MODEL'); model.save('models/$RERANKER_MODEL')"
fi

# 9. Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir data
fi

echo "Setup complete!"
#echo "To start the RAG system, run: source venv/bin/activate && python3 rag_system.py"
echo "To start the RAG system, use run.sh"

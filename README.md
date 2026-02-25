# ANIMA-bot RAG

A Retrieval-Augmented Generation (RAG) system using LlamaIndex, local HuggingFace embeddings, and Ollama.

## Prerequisites

- [Ollama](https://ollama.com/) installed and running.
- A model pulled in Ollama, for example: `ollama pull llama3.1:8b`.
- `BAAI/bge-small-en-v1.5` is downloaded automatically on first run.
- `nomic-embed-text` is optional.

## Setup

1.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Place data (PDFs, text files, etc.) in the `data/` directory.
2.  Run the system:
    ```bash
    python3 rag_system.py
    ```

## Features

- **Local Embeddings**: Uses `BAAI/bge-small-en-v1.5` via `llama-index-embeddings-huggingface` for local embedding generation.
- **Local LLM**: Uses LLMs via Ollama, for example `llama3.1:8b`.
- **Automated Indexing**: Automatically indexes all documents found in the `data/` folder.
- **Index Persistence**: The index is saved to the `storage/` directory; subsequent runs load the index automatically.

### Updating the Index
If you add or remove files in the `data/` directory, delete the `storage/` folder and the index will be rebuilt on the next run:
```bash
rm -rf storage/
./run.sh
```

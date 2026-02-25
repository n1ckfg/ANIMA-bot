# ANIMA-bot RAG System

This project implements a Retrieval-Augmented Generation (RAG) system using LlamaIndex, local HuggingFace embeddings, and Ollama with `llama3.1:8b`.

## Prerequisites

- [Ollama](https://ollama.com/) installed and running.
- `llama3.1:8b` model pulled in Ollama: `ollama pull llama3.1:8b`.
- `nomic-embed-text` is optional (this project uses `BAAI/bge-small-en-v1.5` which is downloaded automatically on first run).

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

1.  Place your data (text files, PDFs, etc.) in the `data/` directory.
2.  Run the RAG system:
    ```bash
    python3 rag_system.py
    ```

## Features

- **Local Embeddings**: Uses `BAAI/bge-small-en-v1.5` via `llama-index-embeddings-huggingface` for high-performance, private embedding generation.
- **Local LLM**: Uses `llama3.1:8b` via Ollama.
- **Automated Indexing**: Automatically indexes all documents found in the `data/` folder.

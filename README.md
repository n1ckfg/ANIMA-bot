# ANIMA-bot RAG

A local RAG (Retrieval-Augmented Generation) system.

## Prerequisites

- [Ollama](https://ollama.com/) installed and running.

## Setup

1.  Run the setup script. The specified LLM and embeddings models will be downloaded automatically if they are missing.

```bash
bash setup.sh
```

## Usage

1.  Place data (PDFs, text files, etc.) in the `data/` directory.
2.  Run the system.

```bash
bash run.sh
```

## Features

- **Local Embeddings:** Uses `BAAI/bge-m3` for embeddings generation.
- **Local LLM**: Uses `llama3.1:8b` via Ollama.
- **Automated Indexing:** Automatically indexes all documents found in the `data/` folder on first run.
- **Index Persistence:** The index is saved to the `storage/` directory; subsequent runs load the index automatically.
- **Updating the Index:** If you add or remove files in the `data/` directory, delete the `storage/` folder and the index will be rebuilt on the next run.

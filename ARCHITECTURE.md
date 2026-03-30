# ANIMA-bot Architecture

This document provides an overview of the architecture for the ANIMA-bot repository, detailing the main components, data flow, and technology stack.

## System Overview

ANIMA-bot implements a Retrieval-Augmented Generation (RAG) system with a web interface built on Flask. It leverages local models for embeddings and reranking to ensure fast offline performance while using Ollama to run the core LLM for generation.

The system is composed of four primary layers:
1. **Frontend / API Layer**
2. **RAG Core Layer**
3. **Data Pipeline**
4. **Environment & Setup**

---

## 1. Frontend / API Layer (`app.py`)

The entry point of the application is a Flask web server. 
- **Web UI:** Serves a user interface (typically from the `templates/` directory).
- **API Endpoint:** Exposes a `/query` POST endpoint for programmatic access.
- **Initialization:** On startup, it initializes the RAG system by loading the vector index into memory, ensuring that subsequent queries are processed quickly.

## 2. RAG Core (`rag_system.py` + `llm_backends.py`)

The heart of the application, built using **LlamaIndex**. It handles the entire lifecycle of the RAG process:

- **Configuration:** All settings are loaded from `config.yaml`, allowing easy customization without code changes.
- **Ingestion & Indexing:** Reads formatted Markdown documents from the `./data/` directory and constructs a `VectorStoreIndex`.
- **Embeddings:** Utilizes the local `BAAI/bge-m3` model (stored in `./models/bge-m3/`) to generate vector embeddings for the documents.
- **Reranking:** To improve the quality of retrieved contexts before they are sent to the LLM, it employs a `FixedSentenceTransformerRerank` using the local `BAAI/bge-reranker-v2-m3` model (stored in `./models/bge-reranker-v2-m3/`).
- **LLM Integration:** Supports multiple backends via `llm_backends.py`:
  - **Ollama** (default): Local Ollama server
  - **llama.cpp**: Direct GGUF model loading via llama-cpp-python
  - **OpenAI-compatible**: Works with OpenAI API, LM Studio, vLLM, etc.
- **Persistence:** Saves the generated vector index to the `./storage/` directory. This allows the system to avoid costly re-indexing on every restart.

## 3. Data Pipeline (`data_prep.sh`)

Data ingestion is separated from the main runtime to ensure clean, structured data is provided to the RAG system.

- **Source:** Raw documents are placed in the `data_prep/` folder.
- **Processing:** The `data_prep.sh` script utilizes **Docling** to parse and convert these raw documents into structured Markdown format.
- **Output:** The resulting Markdown files are saved to the `./data/` directory, organized into subdirectories (e.g., `Anima_md/`, `Digearth_md/`), ready to be indexed by `rag_system.py`.

## 4. Configuration (`config.yaml`)

The system is configured via a YAML file, allowing users to customize the LLM backend and other settings without modifying code:

- **LLM Backend Selection:** Choose between `ollama`, `llamacpp`, or `openai` backends
- **Model Settings:** Configure model names, paths, timeouts, and backend-specific options
- **Embedding & Reranker:** Configure local model paths and parameters
- **Storage Paths:** Customize data and index storage locations

See `config.example.yaml` for full documentation of available options.

## 5. Environment & Setup Scripts

A set of shell scripts automates the setup and execution environment:

- **`setup.sh`:** 
  - Automates the creation of a Python virtual environment (`venv/`).
  - Installs required Python dependencies from `requirements.txt`.
  - Downloads the necessary Ollama LLM models.
  - Downloads the HuggingFace embedding and reranking models directly into the `./models/` directory for local execution.
- **`run.sh` / `run.command`:** 
  - Acts as the main startup script. 
  - Ensures that the Ollama service is running in the background before launching the Flask application (`app.py`).

---

## Data Flow Summary

1. **Ingestion Pipeline:** 
   Raw Documents (`data_prep/`) -> `data_prep.sh` (Docling) -> Structured Markdown (`data/`).
2. **Indexing Pipeline:** 
   Structured Markdown (`data/`) -> `rag_system.py` (LlamaIndex + bge-m3 embeddings) -> Vector Store (`storage/`).
3. **Query Pipeline:** 
   User Query -> `app.py` (Flask `/query`) -> `rag_system.py` (Retrieval) -> Reranking (bge-reranker-v2-m3) -> LLM Generation (Ollama llama3.1) -> Final Response.

---

## Technology Stack

- **Web Framework:** Flask
- **RAG Framework:** LlamaIndex
- **LLM Runner:** Ollama
- **Embeddings & Reranking:** HuggingFace Transformers, Sentence-Transformers (BAAI BGE Models)
- **Data Parsing/Conversion:** Docling
- **Vector Storage:** Local Filesystem (LlamaIndex JSON-based vector store)

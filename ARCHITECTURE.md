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
- **Web UI:** Serves a user interface (from the `templates/` directory) with support for both standard and streaming responses.
- **API Endpoints:**
  - `GET /` - Web interface
  - `GET /health` - Health check endpoint for monitoring system status
  - `POST /query` - Standard query endpoint with optional caching
  - `POST /query/stream` - Server-Sent Events streaming endpoint for real-time responses
  - `POST /cache/clear` - Clear the query cache
  - `POST /reindex` - Trigger a full document reindex
- **Initialization:** On startup, it initializes the RAG system by loading the vector index into memory, ensuring that subsequent queries are processed quickly.

## 2. RAG Core (`rag_system.py` + `llm_backends.py`)

The heart of the application, built using **LlamaIndex**. It handles the entire lifecycle of the RAG process:

### Configuration
All settings are loaded from `config.yaml`, allowing easy customization without code changes.

### Document Processing
- **Ingestion:** Reads formatted Markdown documents from the `./data/` directory and constructs a `VectorStoreIndex`.
- **Chunking:** Configurable document splitting via `SentenceSplitter` or `SemanticSplitterNodeParser` with customizable chunk size and overlap.
- **Sanitization:** Removes surrogate characters from PDF parsing artifacts.

### Embeddings & Retrieval
- **Embeddings:** Utilizes the local `BAAI/bge-m3` model (stored in `./models/bge-m3/`) to generate vector embeddings.
- **Hybrid Search:** Combines BM25 keyword search with vector similarity for improved retrieval:
  - Configurable alpha parameter balances BM25 vs. vector weights
  - Catches keyword-exact matches that pure semantic search might miss
- **Reranking:** Uses `FixedSentenceTransformerRerank` (a custom subclass that fixes tokenizer initialization issues) with the local `BAAI/bge-reranker-v2-m3` model to improve context quality before LLM generation.

### Query Enhancement
- **HyDE (Hypothetical Document Embeddings):** Generates a hypothetical answer passage before retrieval to improve semantic matching.
- **Query Caching:** Disk-based caching via `diskcache` to speed up repeated queries.

### LLM Integration
Supports multiple backends via `llm_backends.py`:
- **Ollama** (default): Local Ollama server
- **llama.cpp**: Direct GGUF model loading via llama-cpp-python
- **OpenAI-compatible**: Works with OpenAI API, LM Studio, vLLM, etc.

### Persistence & Versioning
- **Index Persistence:** Saves the generated vector index to the `./storage/` directory.
- **Index Versioning:** Tracks document hashes in `index_version.json` to detect changes and enable incremental updates.
- **Checkpointing:** During long indexing runs, saves partial progress every 50 documents.

### Error Recovery
- **Timeout Protection:** Documents that exceed 2 hours during embedding are skipped.
- **Retry Logic:** Transient failures are retried up to 3 times before skipping.
- **Progress Tracking:** Real-time progress with ETA during indexing.

## 3. Data Pipeline (`data_prep.sh`)

Data ingestion is separated from the main runtime to ensure clean, structured data is provided to the RAG system.

- **Source:** Raw documents are placed in the `data_prep/` folder.
- **Processing:** The `data_prep.sh` script utilizes **Docling** to parse and convert these raw documents into structured Markdown format.
- **Output:** The resulting Markdown files are saved to the `./data/` directory, organized into subdirectories (e.g., `Anima_md/`, `Digearth_md/`), ready to be indexed by `rag_system.py`.

## 4. Configuration (`config.yaml`)

The system is configured via a YAML file, allowing users to customize all aspects:

### LLM Settings
- **Backend Selection:** Choose between `ollama`, `llamacpp`, or `openai` backends
- **Model Settings:** Configure model names, paths, timeouts, and backend-specific options

### Retrieval Settings
- **similarity_top_k:** Number of candidates to retrieve (default: 25)
- **hybrid_search:** Enable/disable BM25 + vector hybrid search
- **hybrid_alpha:** Balance between vector (1.0) and BM25 (0.0) scoring

### Chunking Settings
- **chunk_size:** Target chunk size in tokens (default: 512)
- **chunk_overlap:** Overlap between chunks (default: 64)
- **splitting_method:** "sentence" or "semantic"

### Query Enhancement
- **enable_hyde:** Enable HyDE query transformation
- **hyde_prompt:** Customizable prompt template for HyDE

### Caching
- **enabled:** Enable/disable query caching
- **cache_dir:** Directory for cache storage

### Storage
- **persist_dir:** Vector index storage directory
- **data_dir:** Source documents directory
- **index_version_file:** Document version tracking file

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
  - Checks and optionally starts the Ollama service (if using ollama backend).
  - Launches the Flask application (`app.py`).

---

## Data Flow Summary

1. **Ingestion Pipeline:** 
   Raw Documents (`data_prep/`) -> `data_prep.sh` (Docling) -> Structured Markdown (`data/`).

2. **Indexing Pipeline:** 
   Structured Markdown (`data/`) -> Chunking (SentenceSplitter) -> `rag_system.py` (LlamaIndex + bge-m3 embeddings) -> Vector Store + BM25 Index (`storage/`).

3. **Query Pipeline:** 
   User Query -> `app.py` (Flask `/query`) -> Query Cache Check -> HyDE Transform (optional) -> Hybrid Retrieval (BM25 + Vector) -> Reranking (bge-reranker-v2-m3) -> LLM Generation (Ollama) -> Cache Store -> Final Response.

---

## Technology Stack

- **Web Framework:** Flask (with SSE streaming support)
- **RAG Framework:** LlamaIndex
- **LLM Runner:** Ollama (default), llama.cpp, or OpenAI-compatible
- **Embeddings & Reranking:** HuggingFace Transformers, Sentence-Transformers (BAAI BGE Models)
- **Hybrid Search:** rank-bm25 for BM25 scoring
- **Caching:** diskcache for disk-based query caching
- **Data Parsing/Conversion:** Docling
- **Vector Storage:** Local Filesystem (LlamaIndex JSON-based vector store)

---

## API Reference

### Health Check
```
GET /health
```
Returns system health status and component states.

### Query
```
POST /query
Content-Type: application/json

{
  "query": "Your question here",
  "use_cache": true
}
```

### Streaming Query
```
POST /query/stream
Content-Type: application/json

{
  "query": "Your question here"
}
```
Returns Server-Sent Events stream.

### Clear Cache
```
POST /cache/clear
```

### Reindex
```
POST /reindex
```
Triggers a full document reindex.

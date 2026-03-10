import os
import sys
import logging
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Suppress logging and warnings from third-party libraries
os.environ["HF_HUB_OFFLINE"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings, 
    StorageContext, 
    load_index_from_storage
)

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def print_progress(current, total, start_time, last_doc_time=None):
    """Print progress report for embeddings generation."""
    elapsed = time.time() - start_time
    avg_time_per_doc = elapsed / current if current > 0 else 0
    remaining = total - current
    eta = avg_time_per_doc * remaining

    progress_pct = (current / total) * 100
    bar_width = 20
    filled = int(bar_width * current / total)
    bar = "=" * filled + ">" + " " * (bar_width - filled - 1) if filled < bar_width else "=" * bar_width

    last_doc_str = f" | Last: {format_time(last_doc_time)}" if last_doc_time is not None else ""
    print(f"\r[{bar}] {progress_pct:5.1f}% | {current}/{total} docs | "
          f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}{last_doc_str}", end="", flush=True)

CHAT_MODEL_PROVIDER = "meta-llama"
CHAT_MODEL = "llama3.1:8b"

DATA_MODEL_PROVIDER = "BAAI"
#DATA_MODEL = "bge-small-en-v1.5"
DATA_MODEL = "bge-m3"

def setup_rag():
    # 1. Setup Ollama LLM
    print("Initializing Ollama LLM (" + CHAT_MODEL + ")...")
    llm = Ollama(model=CHAT_MODEL, request_timeout=360.0)
    
    # 2. Setup Local Embedding Model
    # Now loading from the local ./models directory
    model_path = "./models/" + DATA_MODEL
    if not os.path.exists(model_path):
        print(f"Warning: Local model not found at {model_path}. Falling back to Hub.")
        embed_model = HuggingFaceEmbedding(model_name=DATA_MODEL_PROVIDER + "/" + DATA_MODEL)
    else:
        print(f"Initializing local embedding model from {model_path}...")
        embed_model = HuggingFaceEmbedding(model_name=model_path)
    
    # 3. Configure Global Settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # 4. Persistence setup
    PERSIST_DIR = "./storage"
    
    if not os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
        # 5. Create and Save Index
        if not os.path.exists("./data") or not os.listdir("./data"):
            print("No data found in ./data. Creating a sample file...")
            os.makedirs("./data", exist_ok=True)
            with open("./data/sample.txt", "w") as f:
                f.write("ANIMA-bot is a RAG system using local embeddings and Ollama.")
                
        print("Loading documents from ./data...")
        documents = SimpleDirectoryReader("./data", recursive=True).load_data()
        
        # Sanitize text to remove surrogate characters produced by PDF parsing
        for doc in documents:
            doc.set_content(doc.get_content().encode('utf-8', errors='replace').decode('utf-8'))

        print(f"Creating index from {len(documents)} documents...")
        total_docs = len(documents)
        start_time = time.time()

        # Create index with first document to initialize
        DOC_TIMEOUT = 300  # seconds
        doc_start = time.time()
        index = VectorStoreIndex.from_documents([documents[0]], show_progress=False)
        last_doc_time = time.time() - doc_start
        print_progress(1, total_docs, start_time, last_doc_time)
        skipped_count = 0

        # Add remaining documents with progress tracking and timeout
        for i, doc in enumerate(documents[1:], start=2):
            doc_start = time.time()
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(index.insert, doc)
                    future.result(timeout=DOC_TIMEOUT)
                last_doc_time = time.time() - doc_start
                print_progress(i, total_docs, start_time, last_doc_time)
            except FuturesTimeoutError:
                skipped_count += 1
                print(f"\nSkipped document {i}/{total_docs}: exceeded {DOC_TIMEOUT}s timeout")
                print_progress(i, total_docs, start_time, None)

        # Final progress line
        elapsed = time.time() - start_time
        skipped_msg = f", {skipped_count} skipped" if skipped_count > 0 else ""
        print(f"\nCompleted {total_docs} documents in {format_time(elapsed)} "
              f"(avg: {elapsed/total_docs:.2f}s/doc{skipped_msg})")
        
        print(f"Saving index to {PERSIST_DIR}...")
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # 6. Load existing index
        print(f"Loading existing index from {PERSIST_DIR}...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    
    # 7. Create Query Engine
    return index.as_query_engine()

def main():
    try:
        query_engine = setup_rag()
        
        print("\nRAG system ready! Type 'exit' to quit.")
        while True:
            query = input("\nEnter your query: ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            if not query.strip():
                continue
                
            print("\nSearching and generating response...")
            response = query_engine.query(query)
            print("\nResponse:")
            print("-" * 20)
            print(str(response))
            print("-" * 20)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()

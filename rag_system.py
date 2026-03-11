import os
import sys
import logging
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

#CHAT_MODEL = "qwen3:4b-instruct"
CHAT_MODEL="llama3.1:8b"

DATA_MODEL_PROVIDER = "BAAI"
#DATA_MODEL = "bge-small-en-v1.5"
DATA_MODEL = "bge-m3"

RERANKER_MODEL_PROVIDER = "BAAI"
RERANKER_MODEL = "bge-reranker-v2-m3"
RERANKER_TOP_N = 5  # Number of documents to return after reranking

# Suppress logging and warnings from third-party libraries
os.environ["HF_HUB_OFFLINE"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass
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
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor


class FixedSentenceTransformerRerank(SentenceTransformerRerank):
    def __init__(
        self,
        top_n: int = 2,
        model: str = "cross-encoder/stsb-distilroberta-base",
        device: str = None,
        keep_retrieval_score: bool = False,
        trust_remote_code: bool = True,
    ):
        from sentence_transformers import CrossEncoder
        from llama_index.core.utils import infer_torch_device
        
        device = device or infer_torch_device()
        
        # We call the grandparent's __init__ (BaseNodePostprocessor) 
        # to avoid SentenceTransformerRerank's own __init__ which would 
        # trigger a warning before we can override its _model.
        BaseNodePostprocessor.__init__(
            self,
            top_n=top_n,
            model=model,
            device=device,
            keep_retrieval_score=keep_retrieval_score,
            trust_remote_code=trust_remote_code
        )
        
        # Now we initialize _model with the fix
        self._model = CrossEncoder(
            model,
            max_length=512, # Default as in SentenceTransformerRerank
            device=device,
            trust_remote_code=trust_remote_code,
            tokenizer_kwargs={"fix_mistral_regex": False}
        )

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
    bar_width = 10
    filled = int(bar_width * current / total)
    bar = "=" * filled + ">" + " " * (bar_width - filled - 1) if filled < bar_width else "=" * bar_width

    last_doc_str = f" | Last: {format_time(last_doc_time)}" if last_doc_time is not None else ""
    print(f"\r[{bar}] {progress_pct:5.1f}% | {current}/{total} docs | "
          f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}{last_doc_str}", end="", flush=True)

def setup_reranker():
    """Initialize the reranker model."""
    reranker_path = "./models/" + RERANKER_MODEL
    if not os.path.exists(reranker_path):
        print(f"Warning: Local reranker not found at {reranker_path}.")
        model_name = RERANKER_MODEL_PROVIDER + "/" + RERANKER_MODEL
    else:
        print(f"Initializing local reranker from {reranker_path}...")
        model_name = reranker_path

    return FixedSentenceTransformerRerank(
        model=model_name,
        top_n=RERANKER_TOP_N
    )

def setup_rag():
    # 1. Setup Ollama LLM
    print("Initializing Ollama LLM (" + CHAT_MODEL + ")...")
    llm = Ollama(model=CHAT_MODEL, request_timeout=360.0)
    
    # 2. Setup Local Embedding Model
    # Now loading from the local ./models directory
    model_path = "./models/" + DATA_MODEL
    if not os.path.exists(model_path):
        print(f"Warning: Local model not found at {model_path}.")
        embed_model = HuggingFaceEmbedding(
            model_name=DATA_MODEL_PROVIDER + "/" + DATA_MODEL,
            tokenizer_kwargs={"fix_mistral_regex": False}
        )
    else:
        print(f"Initializing local embedding model from {model_path}...")
        embed_model = HuggingFaceEmbedding(
            model_name=model_path,
            tokenizer_kwargs={"fix_mistral_regex": False}
        )
    
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
        STATUS_INTERVAL = 10  # Print status every 10 seconds for slow documents
        for i, doc in enumerate(documents[1:], start=2):
            doc_start = time.time()
            doc_name = doc.metadata.get('file_name', doc.metadata.get('file_path', f'doc {i}'))
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(index.insert, doc)
                    # Poll in intervals to provide status updates for slow documents
                    while True:
                        try:
                            future.result(timeout=STATUS_INTERVAL)
                            break  # Document completed
                        except FuturesTimeoutError:
                            doc_elapsed = time.time() - doc_start
                            if doc_elapsed >= DOC_TIMEOUT:
                                raise  # Exceeded total timeout
                            print(f"\n  Processing: {doc_name} ({doc_elapsed:.0f}s elapsed...)")
                last_doc_time = time.time() - doc_start
                print_progress(i, total_docs, start_time, last_doc_time)
            except FuturesTimeoutError:
                skipped_count += 1
                print(f"\nSkipped document {i}/{total_docs} ({doc_name}): exceeded {DOC_TIMEOUT}s timeout")
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
    
    # 7. Setup Reranker
    reranker = setup_reranker()

    # 8. Create Query Engine with reranker
    return index.as_query_engine(
        similarity_top_k=10,  # Retrieve more docs initially for reranking
        node_postprocessors=[reranker]
    )

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

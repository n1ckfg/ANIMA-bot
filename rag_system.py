import os
import sys
import logging
import warnings
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from llm_backends import create_llm, get_backend_info

# Default config path
DEFAULT_CONFIG_PATH = "./config.yaml"

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

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}, using defaults...")
        return get_default_config()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_default_config() -> dict:
    """Return default configuration."""
    return {
        "llm": {
            "backend": "ollama",
            "model": "llama3.1:8b",
            "request_timeout": 360,
            "ollama": {"base_url": "http://localhost:11434"},
        },
        "embeddings": {
            "provider": "BAAI",
            "model": "bge-m3",
            "local_path": "./models/bge-m3",
        },
        "reranker": {
            "provider": "BAAI",
            "model": "bge-reranker-v2-m3",
            "local_path": "./models/bge-reranker-v2-m3",
            "top_n": 5,
        },
        "retrieval": {"similarity_top_k": 10},
        "storage": {"persist_dir": "./storage", "data_dir": "./data"},
    }


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


def setup_reranker(config: dict):
    """Initialize the reranker model."""
    reranker_config = config.get("reranker", {})
    provider = reranker_config.get("provider", "BAAI")
    model = reranker_config.get("model", "bge-reranker-v2-m3")
    local_path = reranker_config.get("local_path", f"./models/{model}")
    top_n = reranker_config.get("top_n", 5)

    if os.path.exists(local_path):
        print(f"Initializing local reranker from {local_path}...")
        model_name = local_path
    else:
        print(f"Warning: Local reranker not found at {local_path}.")
        model_name = f"{provider}/{model}"

    return FixedSentenceTransformerRerank(
        model=model_name,
        top_n=top_n
    )


def setup_embeddings(config: dict):
    """Initialize the embedding model."""
    embed_config = config.get("embeddings", {})
    provider = embed_config.get("provider", "BAAI")
    model = embed_config.get("model", "bge-m3")
    local_path = embed_config.get("local_path", f"./models/{model}")

    if os.path.exists(local_path):
        print(f"Initializing local embedding model from {local_path}...")
        model_name = local_path
    else:
        print(f"Warning: Local model not found at {local_path}.")
        model_name = f"{provider}/{model}"

    return HuggingFaceEmbedding(
        model_name=model_name,
        tokenizer_kwargs={"fix_mistral_regex": False}
    )


def setup_rag(config_path: str = DEFAULT_CONFIG_PATH):
    """
    Initialize the RAG system.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Query engine ready for use
    """
    # Load configuration
    config = load_config(config_path)

    # 1. Setup LLM from configured backend
    print(f"Initializing LLM: {get_backend_info(config)}...")
    llm = create_llm(config)

    # 2. Setup Local Embedding Model
    embed_model = setup_embeddings(config)

    # 3. Configure Global Settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    # 4. Persistence setup
    storage_config = config.get("storage", {})
    persist_dir = storage_config.get("persist_dir", "./storage")
    data_dir = storage_config.get("data_dir", "./data")

    if not os.path.exists(os.path.join(persist_dir, "docstore.json")):
        # 5. Create and Save Index
        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            print(f"No data found in {data_dir}. Creating a sample file...")
            os.makedirs(data_dir, exist_ok=True)
            with open(os.path.join(data_dir, "sample.txt"), "w") as f:
                f.write("ANIMA-bot is a RAG system using local embeddings and configurable LLM backends.")

        print(f"Loading documents from {data_dir}...")
        documents = SimpleDirectoryReader(data_dir, recursive=True).load_data()

        # Sanitize text to remove surrogate characters produced by PDF parsing
        for doc in documents:
            doc.set_content(doc.get_content().encode('utf-8', errors='replace').decode('utf-8'))

        print(f"Creating index from {len(documents)} documents...")
        total_docs = len(documents)
        start_time = time.time()

        # Create index with first document to initialize
        DOC_TIMEOUT = 7200  # seconds
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

        print(f"Saving index to {persist_dir}...")
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        # 6. Load existing index
        print(f"Loading existing index from {persist_dir}...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)

    # 7. Setup Reranker
    reranker = setup_reranker(config)

    # 8. Create Query Engine with reranker
    retrieval_config = config.get("retrieval", {})
    similarity_top_k = retrieval_config.get("similarity_top_k", 10)

    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[reranker]
    )


def main():
    try:
        # Allow config path override via command line
        config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
        query_engine = setup_rag(config_path)

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

import os
import sys
import json
import hashlib
import logging
import warnings
import time
import yaml
from pathlib import Path
from typing import List, Optional, Any, Generator
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
    load_index_from_storage,
    Document,
    QueryBundle
)
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.base.base_retriever import BaseRetriever

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor

# BM25 for hybrid search
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("Warning: rank-bm25 not installed. Hybrid search disabled.")

# Disk cache for query caching
try:
    import diskcache
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False
    print("Warning: diskcache not installed. Query caching disabled.")


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
        "retrieval": {
            "similarity_top_k": 25,
            "hybrid_search": True,
            "hybrid_alpha": 0.5,
        },
        "chunking": {
            "chunk_size": 512,
            "chunk_overlap": 64,
            "splitting_method": "sentence",
        },
        "query": {
            "enable_hyde": True,
            "hyde_prompt": "Write a detailed passage that would answer this question: {query}",
        },
        "caching": {
            "enabled": True,
            "cache_dir": "./cache",
        },
        "storage": {
            "persist_dir": "./storage",
            "data_dir": "./data",
            "index_version_file": "./storage/index_version.json",
        },
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
            max_length=512,  # Default as in SentenceTransformerRerank
            device=device,
            trust_remote_code=trust_remote_code,
            tokenizer_kwargs={"fix_mistral_regex": False}
        )


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining BM25 and vector search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        nodes: List[TextNode],
        alpha: float = 0.5,
        top_k: int = 10,
    ):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.alpha = alpha
        self.top_k = top_k

        # Build BM25 index
        if HAS_BM25 and nodes:
            tokenized_corpus = [self._tokenize(node.get_content()) for node in nodes]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.nodes = nodes
        else:
            self.bm25 = None
            self.nodes = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        return text.lower().split()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve using hybrid approach."""
        query = query_bundle.query_str

        # Get vector results
        vector_results = self.vector_retriever.retrieve(query_bundle)

        if not self.bm25 or not self.nodes:
            return vector_results[:self.top_k]

        # Get BM25 results
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Create node ID to BM25 score mapping
        bm25_score_map = {}
        for i, node in enumerate(self.nodes):
            bm25_score_map[node.node_id] = bm25_scores[i]

        # Normalize scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        max_vector = max(r.score for r in vector_results) if vector_results else 1

        # Combine scores
        combined_results = {}
        for result in vector_results:
            node_id = result.node.node_id
            vector_score = result.score / max_vector if max_vector > 0 else 0
            bm25_score = bm25_score_map.get(node_id, 0) / max_bm25
            combined_score = self.alpha * vector_score + (1 - self.alpha) * bm25_score
            combined_results[node_id] = NodeWithScore(node=result.node, score=combined_score)

        # Add BM25-only results that weren't in vector results
        for i, node in enumerate(self.nodes):
            if node.node_id not in combined_results:
                bm25_score = bm25_scores[i] / max_bm25
                if bm25_score > 0.1:  # Only add if BM25 score is meaningful
                    combined_results[node.node_id] = NodeWithScore(
                        node=node,
                        score=(1 - self.alpha) * bm25_score
                    )

        # Sort by combined score and return top_k
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x.score,
            reverse=True
        )
        return sorted_results[:self.top_k]


class QueryCache:
    """Disk-based cache for query results."""

    def __init__(self, cache_dir: str, enabled: bool = True):
        self.enabled = enabled and HAS_CACHE
        if self.enabled:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = diskcache.Cache(cache_dir)
        else:
            self.cache = None

    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        if not self.enabled:
            return None
        key = self._hash_query(query)
        return self.cache.get(key)

    def set(self, query: str, response: str, expire: int = 3600):
        if not self.enabled:
            return
        key = self._hash_query(query)
        self.cache.set(key, response, expire=expire)

    def clear(self):
        if self.enabled:
            self.cache.clear()


class IndexVersionManager:
    """Track indexed documents for incremental updates."""

    def __init__(self, version_file: str):
        self.version_file = version_file
        self.versions = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.version_file):
            with open(self.version_file, "r") as f:
                return json.load(f)
        return {"documents": {}, "last_indexed": None}

    def _save(self):
        os.makedirs(os.path.dirname(self.version_file), exist_ok=True)
        with open(self.version_file, "w") as f:
            json.dump(self.versions, f, indent=2)

    def _hash_file(self, filepath: str) -> str:
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_changes(self, data_dir: str) -> tuple:
        """Returns (new_files, modified_files, deleted_files)."""
        current_files = {}
        for root, _, files in os.walk(data_dir):
            for file in files:
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, data_dir)
                current_files[rel_path] = self._hash_file(filepath)

        indexed = self.versions.get("documents", {})

        new_files = [f for f in current_files if f not in indexed]
        modified_files = [
            f for f in current_files
            if f in indexed and current_files[f] != indexed[f]
        ]
        deleted_files = [f for f in indexed if f not in current_files]

        return new_files, modified_files, deleted_files

    def update(self, data_dir: str):
        """Update version tracking after indexing."""
        self.versions["documents"] = {}
        for root, _, files in os.walk(data_dir):
            for file in files:
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, data_dir)
                self.versions["documents"][rel_path] = self._hash_file(filepath)
        self.versions["last_indexed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save()

    def needs_reindex(self, data_dir: str) -> bool:
        """Check if any files have changed since last index."""
        new, modified, deleted = self.get_changes(data_dir)
        return bool(new or modified or deleted)


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


def setup_node_parser(config: dict, embed_model=None):
    """Initialize the document chunking strategy."""
    chunk_config = config.get("chunking", {})
    chunk_size = chunk_config.get("chunk_size", 512)
    chunk_overlap = chunk_config.get("chunk_overlap", 64)
    method = chunk_config.get("splitting_method", "sentence")

    if method == "semantic" and embed_model:
        print(f"Using semantic chunking with embed model...")
        return SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=1,
            breakpoint_percentile_threshold=95
        )
    else:
        print(f"Using sentence chunking (size={chunk_size}, overlap={chunk_overlap})...")
        return SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )


class HyDEQueryTransform:
    """Hypothetical Document Embeddings for improved retrieval."""

    def __init__(self, llm, prompt_template: str):
        self.llm = llm
        self.prompt_template = prompt_template

    def transform(self, query: str) -> str:
        """Generate a hypothetical document for the query."""
        prompt = self.prompt_template.format(query=query)
        try:
            response = self.llm.complete(prompt)
            # Return both original query and hypothetical for better coverage
            return f"{query}\n\n{response.text}"
        except Exception as e:
            print(f"HyDE transform failed: {e}, using original query")
            return query


class EnhancedRAGSystem:
    """Enhanced RAG system with hybrid search, caching, and HyDE."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        self.llm = None
        self.embed_model = None
        self.index = None
        self.query_engine = None
        self.cache = None
        self.hyde = None
        self.version_manager = None
        self._all_nodes = []  # Store nodes for hybrid search

    def initialize(self):
        """Initialize all components."""
        # 1. Setup LLM
        print(f"Initializing LLM: {get_backend_info(self.config)}...")
        self.llm = create_llm(self.config)

        # 2. Setup Embeddings
        self.embed_model = setup_embeddings(self.config)

        # 3. Configure Global Settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # 4. Setup node parser (chunking)
        node_parser = setup_node_parser(self.config, self.embed_model)
        Settings.node_parser = node_parser

        # 5. Setup caching
        cache_config = self.config.get("caching", {})
        self.cache = QueryCache(
            cache_dir=cache_config.get("cache_dir", "./cache"),
            enabled=cache_config.get("enabled", True)
        )

        # 6. Setup HyDE if enabled
        query_config = self.config.get("query", {})
        if query_config.get("enable_hyde", True):
            print("Initializing HyDE query transformation...")
            self.hyde = HyDEQueryTransform(
                llm=self.llm,
                prompt_template=query_config.get(
                    "hyde_prompt",
                    "Write a detailed passage that would answer this question: {query}"
                )
            )

        # 7. Setup index versioning
        storage_config = self.config.get("storage", {})
        version_file = storage_config.get("index_version_file", "./storage/index_version.json")
        self.version_manager = IndexVersionManager(version_file)

        # 8. Load or create index
        self._setup_index()

        # 9. Create query engine
        self._setup_query_engine()

        return self

    def _setup_index(self):
        """Load existing index or create new one."""
        storage_config = self.config.get("storage", {})
        persist_dir = storage_config.get("persist_dir", "./storage")
        data_dir = storage_config.get("data_dir", "./data")

        index_exists = os.path.exists(os.path.join(persist_dir, "docstore.json"))
        needs_reindex = self.version_manager.needs_reindex(data_dir) if index_exists else True

        if index_exists and not needs_reindex:
            print(f"Loading existing index from {persist_dir}...")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self.index = load_index_from_storage(storage_context)
            # Load nodes for hybrid search
            self._all_nodes = list(self.index.docstore.docs.values())
        else:
            if index_exists:
                new, modified, deleted = self.version_manager.get_changes(data_dir)
                print(f"Index outdated: {len(new)} new, {len(modified)} modified, {len(deleted)} deleted files")
            self._create_index(data_dir, persist_dir)

    def _create_index(self, data_dir: str, persist_dir: str):
        """Create index from documents with progress tracking and error recovery."""
        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            print(f"No data found in {data_dir}. Creating a sample file...")
            os.makedirs(data_dir, exist_ok=True)
            with open(os.path.join(data_dir, "sample.txt"), "w") as f:
                f.write("ANIMA-bot is a RAG system using local embeddings and configurable LLM backends.")

        print(f"Loading documents from {data_dir}...")
        documents = SimpleDirectoryReader(data_dir, recursive=True).load_data()

        # Sanitize text to remove surrogate characters
        for doc in documents:
            doc.set_content(doc.get_content().encode('utf-8', errors='replace').decode('utf-8'))

        print(f"Creating index from {len(documents)} documents...")
        total_docs = len(documents)
        start_time = time.time()

        DOC_TIMEOUT = 7200  # seconds
        STATUS_INTERVAL = 10
        CHECKPOINT_INTERVAL = 50  # Save partial index every N documents

        # Create index with first document
        doc_start = time.time()
        self.index = VectorStoreIndex.from_documents([documents[0]], show_progress=False)
        last_doc_time = time.time() - doc_start
        print_progress(1, total_docs, start_time, last_doc_time)
        skipped_count = 0
        retry_count = 0

        # Add remaining documents with progress, timeout, and checkpointing
        for i, doc in enumerate(documents[1:], start=2):
            doc_start = time.time()
            doc_name = doc.metadata.get('file_name', doc.metadata.get('file_path', f'doc {i}'))

            # Retry logic for transient failures
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(self.index.insert, doc)
                        while True:
                            try:
                                future.result(timeout=STATUS_INTERVAL)
                                break
                            except FuturesTimeoutError:
                                doc_elapsed = time.time() - doc_start
                                if doc_elapsed >= DOC_TIMEOUT:
                                    raise
                                print(f"\n  Processing: {doc_name} ({doc_elapsed:.0f}s elapsed...)")

                    last_doc_time = time.time() - doc_start
                    print_progress(i, total_docs, start_time, last_doc_time)
                    break  # Success, exit retry loop

                except FuturesTimeoutError:
                    skipped_count += 1
                    print(f"\nSkipped document {i}/{total_docs} ({doc_name}): exceeded {DOC_TIMEOUT}s timeout")
                    print_progress(i, total_docs, start_time, None)
                    break  # Don't retry timeouts

                except Exception as e:
                    if attempt < max_retries - 1:
                        retry_count += 1
                        print(f"\nRetrying document {i}/{total_docs} ({doc_name}): {e}")
                        time.sleep(1)  # Brief pause before retry
                    else:
                        skipped_count += 1
                        print(f"\nSkipped document {i}/{total_docs} ({doc_name}) after {max_retries} attempts: {e}")
                        print_progress(i, total_docs, start_time, None)

            # Checkpoint: save partial index periodically
            if i % CHECKPOINT_INTERVAL == 0:
                print(f"\n  Checkpoint: saving partial index ({i}/{total_docs} docs)...")
                self.index.storage_context.persist(persist_dir=persist_dir)

        # Final stats
        elapsed = time.time() - start_time
        skipped_msg = f", {skipped_count} skipped" if skipped_count > 0 else ""
        retry_msg = f", {retry_count} retries" if retry_count > 0 else ""
        print(f"\nCompleted {total_docs} documents in {format_time(elapsed)} "
              f"(avg: {elapsed/total_docs:.2f}s/doc{skipped_msg}{retry_msg})")

        # Save final index
        print(f"Saving index to {persist_dir}...")
        self.index.storage_context.persist(persist_dir=persist_dir)

        # Update version tracking
        self.version_manager.update(self.config.get("storage", {}).get("data_dir", "./data"))

        # Store nodes for hybrid search
        self._all_nodes = list(self.index.docstore.docs.values())

    def _setup_query_engine(self):
        """Setup query engine with hybrid retrieval and reranking."""
        retrieval_config = self.config.get("retrieval", {})
        similarity_top_k = retrieval_config.get("similarity_top_k", 25)
        use_hybrid = retrieval_config.get("hybrid_search", True) and HAS_BM25
        hybrid_alpha = retrieval_config.get("hybrid_alpha", 0.5)

        # Setup reranker
        reranker = setup_reranker(self.config)

        # Setup retriever
        vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k
        )

        if use_hybrid and self._all_nodes:
            print(f"Using hybrid search (alpha={hybrid_alpha})...")
            retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                nodes=self._all_nodes,
                alpha=hybrid_alpha,
                top_k=similarity_top_k
            )
        else:
            print("Using vector-only search...")
            retriever = vector_retriever

        # Create query engine
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[reranker]
        )

    def query(self, query_text: str, use_cache: bool = True) -> str:
        """Query the RAG system with optional caching and HyDE."""
        # Check cache
        if use_cache:
            cached = self.cache.get(query_text)
            if cached:
                return cached

        # Apply HyDE transformation if enabled
        if self.hyde:
            transformed_query = self.hyde.transform(query_text)
        else:
            transformed_query = query_text

        # Execute query
        response = self.query_engine.query(transformed_query)
        result = str(response)

        # Cache result
        if use_cache:
            self.cache.set(query_text, result)

        return result

    def query_stream(self, query_text: str) -> Generator[str, None, None]:
        """Stream query response for real-time output."""
        # Apply HyDE transformation if enabled
        if self.hyde:
            transformed_query = self.hyde.transform(query_text)
        else:
            transformed_query = query_text

        # Execute streaming query
        response = self.query_engine.query(transformed_query)

        # Try to stream if response supports it
        if hasattr(response, 'response_gen'):
            for chunk in response.response_gen:
                yield chunk
        else:
            yield str(response)

    def clear_cache(self):
        """Clear the query cache."""
        self.cache.clear()
        print("Query cache cleared.")

    def reindex(self):
        """Force a full reindex."""
        storage_config = self.config.get("storage", {})
        persist_dir = storage_config.get("persist_dir", "./storage")
        data_dir = storage_config.get("data_dir", "./data")

        # Clear existing index
        import shutil
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        self._create_index(data_dir, persist_dir)
        self._setup_query_engine()
        print("Reindex complete.")


# Backwards compatibility: setup_rag function
def setup_rag(config_path: str = DEFAULT_CONFIG_PATH):
    """
    Initialize the RAG system (backwards compatible).

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Query engine ready for use
    """
    rag = EnhancedRAGSystem(config_path)
    rag.initialize()
    return rag.query_engine


# Global instance for app.py to access advanced features
_rag_instance: Optional[EnhancedRAGSystem] = None


def get_rag_system(config_path: str = DEFAULT_CONFIG_PATH) -> EnhancedRAGSystem:
    """Get or create the global RAG system instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = EnhancedRAGSystem(config_path)
        _rag_instance.initialize()
    return _rag_instance


def main():
    try:
        config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
        rag = get_rag_system(config_path)

        print("\nRAG system ready! Type 'exit' to quit, 'clear' to clear cache, 'reindex' to rebuild index.")
        while True:
            query = input("\nEnter your query: ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            if query.lower() == "clear":
                rag.clear_cache()
                continue
            if query.lower() == "reindex":
                rag.reindex()
                continue

            if not query.strip():
                continue

            print("\nSearching and generating response...")
            response = rag.query(query)
            print("\nResponse:")
            print("-" * 20)
            print(response)
            print("-" * 20)

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()

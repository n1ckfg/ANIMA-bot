import os
import sys
import logging
import warnings

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

def setup_rag():
    # 1. Setup Ollama LLM
    print("Initializing Ollama LLM (llama3.1:8b)...")
    llm = Ollama(model="llama3.1:8b", request_timeout=360.0)
    
    # 2. Setup Local Embedding Model
    # Now loading from the local ./models directory
    model_path = "./models/bge-small-en-v1.5"
    if not os.path.exists(model_path):
        print(f"Warning: Local model not found at {model_path}. Falling back to Hub.")
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    else:
        print(f"Initializing Local Embedding Model from {model_path}...")
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
                f.write("ANIMA-bot is an advanced autonomous agent system for M2 Max Macbooks. It uses local embeddings and Ollama to process information efficiently.")
                
        print("Loading documents from ./data...")
        documents = SimpleDirectoryReader("./data").load_data()
        
        # Sanitize text to remove surrogate characters produced by PDF parsing
        for doc in documents:
            doc.set_content(doc.get_content().encode('utf-8', errors='replace').decode('utf-8'))

        print(f"Creating index from {len(documents)} documents...")
        index = VectorStoreIndex.from_documents(documents)
        
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
        
        print("\nRAG System Ready! Type 'exit' to quit.")
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

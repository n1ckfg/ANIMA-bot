import os
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def setup_rag():
    # 1. Setup Ollama LLM
    # llama3.1:8b is already installed on the system
    print("Initializing Ollama LLM (llama3.1:8b)...")
    llm = Ollama(model="llama3.1:8b", request_timeout=360.0)
    
    # 2. Setup Local Embedding Model
    # BAAI/bge-small-en-v1.5 is a great local model for high-performance retrieval
    print("Initializing Local Embedding Model (BAAI/bge-small-en-v1.5)...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 3. Configure Global Settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # 4. Load Data
    if not os.path.exists("./data") or not os.listdir("./data"):
        print("No data found in ./data. Creating a sample file...")
        os.makedirs("./data", exist_ok=True)
        with open("./data/sample.txt", "w") as f:
            f.write("ANIMA-bot is an advanced autonomous agent system for M2 Max Macbooks. It uses local embeddings and Ollama to process information efficiently.")
            
    print("Loading documents from ./data...")
    documents = SimpleDirectoryReader("./data").load_data()
    
    # 5. Create Index
    print(f"Creating index from {len(documents)} documents...")
    index = VectorStoreIndex.from_documents(documents)
    
    # 6. Create Query Engine
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

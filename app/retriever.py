"""
Retriever for Legal Cases Vector Database
"""
import chromadb
from pathlib import Path
from typing import List, Dict
import sys
import os
from openai import OpenAI
from dotenv import load_dotenv


class LegalCaseRetriever:
    
    def __init__(self, db_path: str, api_key: str):

        self.db_path = Path(db_path)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at: {self.db_path}")
        
        self.openai_client = OpenAI(api_key=api_key)
        self.embedding_model = "text-embedding-3-small"
        print(f"Loading vector database from: {self.db_path}")
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.chroma_client.get_collection(name="legal_cases")
    
        print(f"Loaded collection with {self.collection.count()} vectors\n")
    
    def generate_query_embedding(self, query: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        return response.data[0].embedding
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k most relevant chunks for a query."""
        query_embedding = self.generate_query_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        retrieved = []
        for i, (doc, metadata, distance) in enumerate(
            zip(results['documents'][0], results['metadatas'][0], results['distances'][0])
        ):
            retrieved.append({
                'rank': i + 1,
                'relevance_score': 1 - distance,
                'text': doc,
                'source_file': metadata['source_file'],
                'chunk_id': metadata['chunk_id'],
                'token_count': metadata['token_count'],
                'distance': distance
            })
        
        return retrieved
    
    def display_results(self, results: List[Dict], show_full_text: bool = False):

        for result in results:
            print(f"\n[Rank {result['rank']}] "
                  f"Relevance: {result['relevance_score']:.4f}")
            print(f"Source: {result['source_file']}")
            print(f"Chunk: {result['chunk_id']} ({result['token_count']} tokens)")
            
            if show_full_text:
                print(result['text'])
            else:
                preview = result['text'][:400]
                print(preview + ("..." if len(result['text']) > 400 else ""))
            
            print("-" * 40)
    
    def test_query(self, query: str, top_k: int = 5, show_full_text: bool = False):
        print(f"\nQuery: '{query}'")
        print(f"Retrieving top {top_k} relevant chunks...\n")
        
        results = self.retrieve(query, top_k)
        self.display_results(results, show_full_text)
        
        return results


def run_interactive_mode():
    project_root = Path(__file__).parent.parent
    db_path = project_root / "app" / "data" / "vector-db"
    
    load_dotenv(project_root / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        api_key = api_key.strip().strip('"\'')
    
    if not api_key:
        print("Error: OpenAI API key not found in .env file.")
        return
    
    try:
        retriever = LegalCaseRetriever(str(db_path), api_key)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run generate_embeddings.py first to create the vector database.")
        return
    
    print("-" * 80)
    print("CASE RETRIEVER")
    print("-" * 80)
    print("Enter queries to search legal cases.")
    print("Commands: 'quit', 'exit', or 'full' to toggle full text.")
    print("-" * 80)
    
    show_full = False
    
    while True:
        try:
            query = input("\nEnter query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Exiting.")
                break
            
            if query.lower() == 'full':
                show_full = not show_full
                print(f"Full text display: {'ON' if show_full else 'OFF'}")
                continue
            
            if not query:
                continue
            
            top_k_input = input("Number of results (default 5): ").strip()
            top_k = int(top_k_input) if top_k_input.isdigit() else 5
            
            retriever.test_query(query, top_k, show_full)
            
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def run_batch_tests():
    """Run predefined test queries."""
    project_root = Path(__file__).parent.parent
    db_path = project_root / "app" / "data" / "vector-db"
    
    load_dotenv(project_root / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        retriever = LegalCaseRetriever(str(db_path), api_key)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    test_queries = [
        "contract breach damages",
        "intellectual property rights",
        "employment discrimination",
        "criminal liability intent"
    ]
    
    print("\nRUNNING BATCH TESTS")
    print("-" * 80)
    
    for query in test_queries:
        retriever.test_query(query, top_k=3, show_full_text=False)


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        run_batch_tests()
    else:
        run_interactive_mode()


if __name__ == "__main__":
    main()
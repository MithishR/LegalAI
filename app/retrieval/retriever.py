"""
Retriever for Legal Cases Vector Database
"""
import chromadb
from pathlib import Path
from typing import List, Dict, Literal
import sys
import os
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import numpy as np


class LegalCaseRetriever:
    
    def __init__(self, db_path: str, api_key: str, enable_bm25: bool = True):

        self.db_path = Path(db_path)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at: {self.db_path}")
        
        self.openai_client = OpenAI(api_key=api_key)
        self.embedding_model = "text-embedding-3-small"
        print(f"Loading vector database from: {self.db_path}")
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.chroma_client.get_collection(name="legal_cases")
    
        print(f"Loaded collection with {self.collection.count()} vectors")
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_metadata = []
        
        if enable_bm25:
            self._build_bm25_index()
        else:
            print()
    
    def _build_bm25_index(self):
        all_data = self.collection.get(
            include=['documents', 'metadatas']
        )
        
        self.bm25_documents = all_data['documents']
        self.bm25_metadata = all_data['metadatas']
        tokenized_docs = [doc.lower().split() for doc in self.bm25_documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
    
    def generate_query_embedding(self, query: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        return response.data[0].embedding
    
    def retrieve(self, query: str, top_k: int = 5, method: Literal['vector', 'bm25', 'hybrid'] = 'hybrid', alpha: float = 0.5) -> List[Dict]:
        """Retrieve top-k most relevant chunks for a query. """
        if method == 'vector':
            return self._retrieve_vector(query, top_k)
        elif method == 'bm25':
            return self._retrieve_bm25(query, top_k)
        elif method == 'hybrid':
            return self._retrieve_hybrid(query, top_k, alpha)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
    
    def _retrieve_vector(self, query: str, top_k: int = 5) -> List[Dict]:
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
                'relevance_score': 1 / (1 + distance),
                'text': doc,
                'source_file': metadata['source_file'],
                'chunk_id': metadata['chunk_id'],
                'token_count': metadata['token_count'],
                'distance': distance,
                'method': 'vector'
            })
        
        return retrieved
    
    def _retrieve_bm25(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve using BM25 sparse search only."""
        if self.bm25_index is None:
            raise ValueError("BM25 index not initialized. Set enable_bm25=True when creating retriever.")
        
        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        retrieved = []
        for i, idx in enumerate(top_indices):
            metadata = self.bm25_metadata[idx]
            retrieved.append({
                'rank': i + 1,
                'relevance_score': float(scores[idx]),
                'text': self.bm25_documents[idx],
                'source_file': metadata['source_file'],
                'chunk_id': metadata['chunk_id'],
                'token_count': metadata['token_count'],
                'bm25_score': float(scores[idx]),
                'method': 'bm25'
            })
        
        return retrieved
    
    def _retrieve_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
        if self.bm25_index is None:
            raise ValueError("BM25 index not initialized. Set enable_bm25=True when creating retriever.")
        fetch_k = min(top_k * 3, len(self.bm25_documents))
        query_embedding = self.generate_query_embedding(query)
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=['documents', 'metadatas', 'distances']
        )
        query_tokens = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:fetch_k]
        doc_scores = {}
        for i, (doc, metadata, distance) in enumerate(
            zip(vector_results['documents'][0], vector_results['metadatas'][0], vector_results['distances'][0])
        ):
            doc_key = (metadata['source_file'], metadata['chunk_id'])
            vector_score = 1 / (1 + distance)
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {
                    'text': doc,
                    'metadata': metadata,
                    'vector_score': vector_score,
                    'vector_rank': i + 1,
                    'bm25_score': 0,
                    'bm25_rank': None
                }
            else:
                doc_scores[doc_key]['vector_score'] = vector_score
                doc_scores[doc_key]['vector_rank'] = i + 1
        for i, idx in enumerate(bm25_top_indices):
            metadata = self.bm25_metadata[idx]
            doc_key = (metadata['source_file'], metadata['chunk_id'])
            bm25_score = float(bm25_scores[idx])
            
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {
                    'text': self.bm25_documents[idx],
                    'metadata': metadata,
                    'vector_score': 0,
                    'vector_rank': None,
                    'bm25_score': bm25_score,
                    'bm25_rank': i + 1
                }
            else:
                doc_scores[doc_key]['bm25_score'] = bm25_score
                doc_scores[doc_key]['bm25_rank'] = i + 1
        k_rrf = 60  
        for doc_key, doc_info in doc_scores.items():
            rrf_vector = 1 / (k_rrf + doc_info['vector_rank']) if doc_info['vector_rank'] else 0
            rrf_bm25 = 1 / (k_rrf + doc_info['bm25_rank']) if doc_info['bm25_rank'] else 0
            doc_info['hybrid_score'] = alpha * rrf_vector + (1 - alpha) * rrf_bm25
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]['hybrid_score'], reverse=True)[:top_k]
        retrieved = []
        for i, (doc_key, doc_info) in enumerate(sorted_docs):
            retrieved.append({
                'rank': i + 1,
                'relevance_score': doc_info['hybrid_score'],
                'text': doc_info['text'],
                'source_file': doc_info['metadata']['source_file'],
                'chunk_id': doc_info['metadata']['chunk_id'],
                'token_count': doc_info['metadata']['token_count'],
                'vector_score': doc_info['vector_score'],
                'bm25_score': doc_info['bm25_score'],
                'method': 'hybrid'
            })
        
        return retrieved
    
    def retrieve_by_source(self, source_files: List[str]) -> List[Dict]:
        all_chunks = []
        
        for source_file in source_files:
            results = self.collection.get(
                where={"source_file": source_file},
                include=['documents', 'metadatas']
            )
            
            if results['documents']:
                for doc, metadata in zip(results['documents'], results['metadatas']):
                    all_chunks.append({
                        'rank': 0,  #
                        'relevance_score': 1.0,  
                        'text': doc,
                        'source_file': metadata['source_file'],
                        'chunk_id': metadata['chunk_id'],
                        'token_count': metadata['token_count'],
                        'distance': 0.0
                    })
    
        all_chunks.sort(key=lambda x: (x['source_file'], x['chunk_id']))
        
        return all_chunks
    
    def display_results(self, results: List[Dict], show_full_text: bool = False):

        for result in results:
            method_info = f" [{result.get('method', 'unknown')}]"
            print(f"\n[Rank {result['rank']}]{method_info} "
                  f"Relevance: {result['relevance_score']:.4f}")
            if result.get('method') == 'hybrid':
                print(f"  Vector: {result.get('vector_score', 0):.4f} | BM25: {result.get('bm25_score', 0):.4f}")
            
            print(f"Source: {result['source_file']}")
            print(f"Chunk: {result['chunk_id']} ({result['token_count']} tokens)")
            
            if show_full_text:
                print(result['text'])
            else:
                preview = result['text'][:400]
                print(preview + ("..." if len(result['text']) > 400 else ""))
            
            print("-" * 40)
    
    def test_query(self, query: str, top_k: int = 5, show_full_text: bool = False, method: str = 'hybrid', alpha: float = 0.5):
        print(f"\nQuery: '{query}'")
        print(f"Method: {method.upper()}" + (f" (alpha={alpha})" if method == 'hybrid' else ""))
        print(f"Retrieving top {top_k} relevant chunks...\n")
        
        results = self.retrieve(query, top_k, method=method, alpha=alpha)
        self.display_results(results, show_full_text)
        
        return results

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



if __name__ == "__main__":
    main()
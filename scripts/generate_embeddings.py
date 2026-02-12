"""
Embedding Generator and Vector Database Storage Script"""
import json
import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import logging
from typing import List, Dict
import time
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings and store in ChromaDB vector database."""
    
    def __init__(self, chunks_dir: str, db_path: str, api_key: str):
        """
        Initialize the embedding generator.
        """
        self.chunks_dir = Path(chunks_dir)
        self.db_path = Path(db_path)
        
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = "text-embedding-3-small"
        
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="legal_cases",
            metadata={"description": "Legal case document embeddings"}
        )
        
        logger.info(f"Collection 'legal_cases' ready. Current count: {self.collection.count()}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI API.

        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def load_chunks_from_file(self, json_file: Path) -> List[Dict]:
        """
        Load chunks from a JSON file.
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            return chunks
        except Exception as e:
            logger.error(f"Error loading {json_file.name}: {str(e)}")
            return []
    
    def process_chunks_batch(self, chunks: List[Dict], batch_size: int = 30):

        
        # we are using 200k tokens/min for each batch. each batch max 30k tokens thus 6 batches/min
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            total_tokens = sum(chunk['token_count'] for chunk in batch)
            logger.info(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} chunks, ~{total_tokens} tokens)")
            
            try:
                texts = [chunk['text'] for chunk in batch]
                
                embeddings = self.generate_embeddings(texts)
                
                ids = []
                metadatas = []
                documents = []
                
                for chunk, embedding in zip(batch, embeddings):
                    chunk_id = f"{chunk['source_file']}_chunk_{chunk['chunk_id']}"
                    ids.append(chunk_id)
                    metadata = {
                        "source_file": chunk['source_file'],
                        "chunk_id": chunk['chunk_id'],
                        "token_count": chunk['token_count'],
                        "start_token": chunk['start_token'],
                        "end_token": chunk['end_token'],
                        "total_tokens_in_source": chunk['total_tokens_in_source']
                    }
                    metadatas.append(metadata)
                    documents.append(chunk['text'])
        
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
                
             
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"  Error processing batch: {str(e)}")
                raise
    
    def process_all_chunk_files(self) -> dict:
        """
        Process all JSON chunk files in the input directory.
        """
        if not self.chunks_dir.exists():
            logger.error(f"Chunks directory does not exist: {self.chunks_dir}")
            return {"files_processed": 0, "total_chunks": 0, "failed": 0}
        
        json_files = list(self.chunks_dir.glob("*_chunks.json"))
        
        if not json_files:
            logger.warning(f"No chunk files found in {self.chunks_dir}")
            return {"files_processed": 0, "total_chunks": 0, "failed": 0}
        
        logger.info(f"Found {len(json_files)} chunk files to process")
        
        files_processed = 0
        total_chunks_processed = 0
        failed = 0
        
        for json_file in json_files:
            try:
                logger.info(f"\nProcessing: {json_file.name}")
                
          
                chunks = self.load_chunks_from_file(json_file)
                
                if not chunks:
                    logger.warning(f"  ✗ No chunks found in {json_file.name}")
                    failed += 1
                    continue
                
                logger.info(f"  Loaded {len(chunks)} chunks")

                self.process_chunks_batch(chunks, batch_size=30)
                
                files_processed += 1
                total_chunks_processed += len(chunks)
                
                logger.info(f"  Completed {json_file.name}")
                
            except Exception as e:
                logger.error(f"  Error processing {json_file.name}: {str(e)}")
                failed += 1
        
        return {
            "files_processed": files_processed,
            "total_chunks": total_chunks_processed,
            "failed": failed,
            "total_files": len(json_files)
        }


def main():
    
    project_root = Path(__file__).parent.parent
    chunks_dir = project_root / "app" / "data" / "chunked-cases"
    db_path = project_root / "app" / "data" / "vector-db"

    load_dotenv(project_root / ".env")
    
    api_key = None
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        api_key = api_key.strip().strip('"\'')  
    if not api_key:
        config_file = project_root / "app" / "config.py"
        if config_file.exists():
            with open(config_file, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'OPENAI_API_KEY' in line and '=' in line:
                        api_key = line.split('=')[1].strip().strip('"\'')
                        break
    
    if not api_key:
        logger.error("OpenAI API key not found!")
        logger.error("Please set OPENAI_API_KEY in:")
        logger.error("  1. Environment variable, OR")
        logger.error("  2. .env file in project root, OR")
        logger.error("  3. app/config.py file")
        return
    

    logger.info("Embedding Generation and Vector Storage Started")
    logger.info(f"Input directory: {chunks_dir}")
    logger.info(f"Vector DB path: {db_path}")
    logger.info(f"Embedding model: text-embedding-3-small")

    
    embedder = EmbeddingGenerator(
        str(chunks_dir),
        str(db_path),
        api_key
    )
    
    stats = embedder.process_all_chunk_files()
    
    final_count = embedder.collection.count()
    
    logger.info("Embedding Generation Complete!")
    logger.info(f"Total chunk files: {stats['total_files']}")
    logger.info(f"Successfully processed: {stats['files_processed']}")
    logger.info(f"Total chunks embedded: {stats['total_chunks']}")
    logger.info(f"Failed files: {stats['failed']}")
    logger.info(f"Total vectors in database: {final_count}")



if __name__ == "__main__":
    main()

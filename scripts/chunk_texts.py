"""
Text Chunking Script-Chunks extracted text files into 1000-token segments for embedding preparation
"""
import json
from pathlib import Path
import tiktoken
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextChunker:
    """Chunk text documents into fixed-token segments."""
    
    def __init__(self, input_dir: str, output_dir: str, chunk_size: int = 1000, overlap: int = 100):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def chunk_text(self, text: str, source_file: str) -> list:
        """
        Chunk a single text document into fixed-token segments.
        """
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        
        logger.info(f"  Total tokens in document: {total_tokens}")
        
        chunks = []
        chunk_index = 0
        
        start_pos = 0
        while start_pos < total_tokens:
            end_pos = min(start_pos + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start_pos:end_pos]
            
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_data = {
                "chunk_id": chunk_index,
                "source_file": source_file,
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "start_token": start_pos,
                "end_token": end_pos,
                "total_tokens_in_source": total_tokens
            }
            
            chunks.append(chunk_data)
            chunk_index += 1
            
            start_pos += (self.chunk_size - self.overlap)
        
        return chunks
    
    def process_all_texts(self) -> dict:
        """
        Process all text files in the input directory.
        """
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return {"files_processed": 0, "total_chunks": 0, "failed": 0}
        
        text_files = list(self.input_dir.glob("*.txt"))
        
        if not text_files:
            logger.warning(f"No text files found in {self.input_dir}")
            return {"files_processed": 0, "total_chunks": 0, "failed": 0}
        
        logger.info(f"Found {len(text_files)} text files to chunk")
        
        files_processed = 0
        total_chunks = 0
        failed = 0
        
        for text_file in text_files:
            try:
                logger.info(f"\nProcessing: {text_file.name}")
                
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                if not text.strip():
                    logger.warning(f"  ✗ Empty file: {text_file.name}")
                    failed += 1
                    continue
                
                chunks = self.chunk_text(text, text_file.name)
                
                output_filename = text_file.stem + "_chunks.json"
                output_path = self.output_dir / output_filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, indent=2, ensure_ascii=False)
                
                logger.info(f"  ✓ Created {len(chunks)} chunks -> {output_filename}")
                
                files_processed += 1
                total_chunks += len(chunks)
                
            except Exception as e:
                logger.error(f"  ✗ Error processing {text_file.name}: {str(e)}")
                failed += 1
        
        return {
            "files_processed": files_processed,
            "total_chunks": total_chunks,
            "failed": failed,
            "total_files": len(text_files)
        }


def main():
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "app" / "data" / "raw-case-txts"
    output_dir = project_root / "app" / "data" / "chunked-cases"
    
    logger.info("="*60)
    logger.info("Text Chunking Process Started")
    logger.info("="*60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Chunk size: 1000 tokens")
    logger.info(f"Overlap: 100 tokens")
    logger.info("="*60)
    
    chunker = TextChunker(
        str(input_dir),
        str(output_dir),
        chunk_size=1000,
        overlap=100
    )
    
    stats = chunker.process_all_texts()
    
    logger.info("\n" + "="*60)
    logger.info("Chunking Complete!")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Successfully processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

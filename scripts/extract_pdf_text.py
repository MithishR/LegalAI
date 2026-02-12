import os
from pathlib import Path
import PyPDF2
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFTextExtractor:
    def __init__(self, pdf_dir: str, output_dir: str):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        text_content = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                logger.info(f"Processing {pdf_path.name} ({num_pages} pages)")
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text:
                        text_content.append(f"--- Page {page_num + 1} ---\n")
                        text_content.append(text)
                        text_content.append("\n\n")
                
            return "".join(text_content)
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path.name}: {str(e)}")
            return ""
    
    def process_all_pdfs(self) -> dict:
  
        if not self.pdf_dir.exists():
            logger.error(f"PDF directory does not exist: {self.pdf_dir}")
            return {"success": 0, "failed": 0, "total": 0}
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_dir}")
            return {"success": 0, "failed": 0, "total": 0}
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        success_count = 0
        failed_count = 0
        
        for pdf_path in pdf_files:
            try:
                text = self.extract_text_from_pdf(pdf_path)
                
                if text.strip():
                    output_filename = pdf_path.stem + ".txt"
                    output_path = self.output_dir / output_filename
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    logger.info(f"✓ Successfully extracted: {pdf_path.name} -> {output_filename}")
                    success_count += 1
                else:
                    logger.warning(f"✗ No text extracted from: {pdf_path.name}")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"✗ Failed to process {pdf_path.name}: {str(e)}")
                failed_count += 1
        
        return {
            "success": success_count,
            "failed": failed_count,
            "total": len(pdf_files)
        }


def main():
    project_root = Path(__file__).parent.parent
    pdf_dir = project_root / "app" / "data" / "case-pdfs"
    output_dir = project_root / "app" / "data" / "raw-case-txts"
    logger.info("PDF Text Extraction Started")
    logger.info(f"Input directory: {pdf_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    extractor = PDFTextExtractor(str(pdf_dir), str(output_dir))
    stats = extractor.process_all_pdfs()
    logger.info("="*60)
    logger.info("Extraction Complete!")
    logger.info(f"Total PDFs: {stats['total']}")
    logger.info(f"Successfully extracted: {stats['success']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

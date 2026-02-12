# LegalAI 

A Retrieval-Augmented Generation (RAG) system for Canadian case law analysis. Ask natural language questions about Supreme Court and CanLII decisions and receive grounded, citation-backed answers.

## Features

- **Semantic Search**: Vector-based retrieval using OpenAI embeddings (1536D)
- **Intelligent Case Detection**: Automatically detects and prioritizes specific cases mentioned in queries
- **CanLII Citation Formatting**: Proper legal citation format for all retrieved cases
- **Token Budget Management**: Smart context fitting (max 4000 tokens)
- **Source Attribution**: All answers include case citations and relevance scores
- **Interactive CLI**: Ask questions in natural language with real-time retrieval

**Technology Stack:**
- **Vector Database**: ChromaDB with persistent storage
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **LLM**: OpenAI `gpt-4o-mini` for answer generation
- **Chunking**: tiktoken with cl100k_base encoding
- **Retrieval**: Euclidean distance with HNSW indexing

## Installation

### Prerequisites
- Python 3.10+
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd LegalAI
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Quick Start

Run the interactive Q&A system:
```bash
python app/agent/rag_pipeline.py
```

**Example queries:**
- "What is the standard for grossly disproportionate sentences?"
- "What was R. v. Vrbanic about?"
- "What is the most recent Supreme Court judgment in 2025?"
- "Explain the test for unreasonable search and seizure"

**Features:**
- Case name detection: "R. v. [Name]" → retrieves all chunks from that case
- Citation detection: "2025 SCC 38" → prioritizes that specific case
- General queries: Semantic search across all cases

### Data Ingestion Pipeline

#### 1. Extract PDF Text
Place PDF files in `app/data/case-pdfs/`:
```bash
python scripts/extract_pdf_text.py
```
Output: `app/data/raw-case-txts/*.txt`

#### 2. Chunk Text
Split text into 1000-token chunks with 100-token overlap:
```bash
python scripts/chunk_texts.py
```
Output: `app/data/chunked-cases/*_chunks.json`

#### 3. Generate Embeddings
Create vector embeddings and store in ChromaDB:
```bash
python scripts/generate_embeddings.py
```
 Output: `app/data/vector-db/`

**Note**: Steps 1-3 must be run sequentially when adding new cases.

## Technical Details

### Chunking Algorithm
- **Strategy**: Sliding window with overlap
- **Chunk size**: 1000 tokens (cl100k_base encoding)
- **Overlap**: 100 tokens between consecutive chunks
- **Purpose**: Preserve context across chunk boundaries

### Embedding & Retrieval
- **Model**: `text-embedding-3-small` (1536 dimensions)
- **Distance metric**: Euclidean distance
- **Relevance score**: `1 / (1 + distance)` (0-1 range, higher = better)
- **Min threshold**: 0.3 relevance score
- **Indexing**: HNSW (Hierarchical Navigable Small World)

### Case-Specific Retrieval
When a query mentions a specific case (by name or citation):
1. **Detect citation**: Regex patterns for "R. v. [Name]" or "YYYY SCC/CanLII NN"
2. **Retrieve all chunks**: Get ALL chunks from that case file (not just top-k)
3. **Prioritize**: Place case-specific chunks first
4. **Context addition**: Fill remaining slots with semantically similar chunks

### Context Management
- **Max tokens**: 4000 (configurable)
- **Budget fitting**: Greedy algorithm, prioritizes highest relevance
- **Metadata**: Each chunk includes source file, citation, chunk ID, and relevance score

### CanLII Citation Format
Filenames are automatically converted to proper legal citations:
- `2025scc38.txt` → "2025 SCC 38 (CanLII)"
- `2025canlii125773.txt` → "2025 CanLII 125773"

## Example Output

```
Your question: What is the standard for grossly disproportionate sentences?

Retrieving relevant case law for: "What is the standard for grossly disproportionate sentences?"
Sources: 2023 SCC 26 (CanLII), 2024 SCC 38 (CanLII), 2025 SCC 41 (CanLII)

ANSWER:

The standard for 'grossly disproportionate' sentences is described as being "so excessive 
as to outrage standards of decency" and "abhorrent or intolerable" to society. The standard 
is intended to be demanding and will be attained only rarely. To establish that a sentence 
is grossly disproportionate, it must be shown that the minimum sentence goes beyond what 
is necessary to achieve Parliament's objectives relevant to the offence.

Sources (3 cases):
  - 2023 SCC 26 (CanLII)
  - 2024 SCC 38 (CanLII)
  - 2025 SCC 41 (CanLII)
Context: 4 chunks, 3847 tokens
Model: gpt-4o-mini-2024-07-18 | Tokens: 4207
```

## Known Limitations

### Data Completeness
- **Oral judgments**: Some recent cases (e.g., R. v. Vrbanic, 2025 CanLII 125773) contain only oral summaries with "Reasons to follow"
- **Solution**: Manually download full written reasons from CanLII when published and re-run ingestion pipeline

### Context Window
- Limited to 4000 tokens (~3-5 pages of case law)
- Very long cases may be truncated
- Complex multi-case questions may lack full context

### Retrieval Accuracy
- Semantic search may occasionally miss relevant chunks
- Requires well-formed natural language queries
- Legal terminology recognition depends on training data

### Not Legal Advice
- **This system summarizes case law contextually**
- Does not provide legal advice or opinions
- Users must verify all information independently
- Consult a qualified lawyer for legal decisions

## Legal & Compliance

### CanLII Terms of Use
- Individual case downloads for research: **Permitted**
- Academic and research use with attribution: **Permitted**
- Bulk downloading or systematic scraping: **Prohibited**
- Commercial use without permission: **Prohibited**

### Attribution
All cases sourced from [CanLII.org](https://www.canlii.org/) - Canadian Legal Information Institute.




## Acknowledgments

- **CanLII** for providing free access to Canadian case law
- **OpenAI** for embedding and LLM APIs
- **ChromaDB** for vector database infrastructure

---

**Disclaimer**: This tool is for research and educational purposes only. It does not constitute legal advice. Always consult a qualified legal professional for legal matters.

---
Made with ❤️ by Mithish :)
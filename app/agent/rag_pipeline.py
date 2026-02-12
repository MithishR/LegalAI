"""
RAG Pipeline - Retrieval Augmented Generation for Legal Case Analysis
Connects retrieval to reasoning with grounded prompts
"""
from pathlib import Path
from typing import List, Dict, Optional
import sys
import os
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from retriever import LegalCaseRetriever
from models.llm_provider import LLMProvider
import re


def parse_canlii_citation(filename: str) -> str:
    name = filename.replace('.txt', '')
    
    # Pattern: YYYY + court code + case number
    match = re.match(r'(\d{4})([a-z]+)(\d+)', name.lower())
    
    if not match:
        return filename  
    
    year, court, number = match.groups()
    
    court_upper = court.upper()
    
    if court_upper == 'CANLII':
        return f"{year} CanLII {number}"
    else:
        return f"{year} {court_upper} {number} (CanLII)"


def extract_case_citations_from_query(query: str) -> List[str]:
    # Pattern 1: "YYYY SCC NN" or "YYYY CanLII NNNNN"
    pattern1 = r'(\d{4})\s+(SCC|CanLII)\s+(\d+)'
    matches1 = re.findall(pattern1, query, re.IGNORECASE)
    
    filenames = []
    for year, court, number in matches1:
        filename = f"{year}{court.lower()}{number}.txt"
        filenames.append(filename)
    
    return filenames


def extract_case_names_from_query(query: str) -> List[str]:
    # Pattern: R. v. [Name] or R v [Name] or R. v [Name]
    pattern = r'R\.?\s*v\.?\s+([A-Z][a-zA-Z]+(?:\s+(?:and|et)\s+[A-Z][a-zA-Z]+)?)'
    matches = re.findall(pattern, query)
    
    case_names = []
    for name in matches:
        case_names.append(f"R. v. {name}")
    
    return case_names


def citation_to_filename(citation: str) -> str:
    citation = re.sub(r'\s*\(CanLII\)\s*$', '', citation)
    match = re.match(r'(\d{4})\s+([A-Za-z]+)\s+(\d+)', citation)
    if match:
        year, court, number = match.groups()
        return f"{year}{court.lower()}{number}.txt"
    
    return None


@dataclass
class RAGContext:
    query: str
    retrieved_chunks: List[Dict]
    total_chunks: int
    sources: List[str]
    total_tokens: int
    
    def get_source_citations(self) -> str:
        unique_sources = list(set(self.sources))
        citations = [parse_canlii_citation(src) for src in sorted(unique_sources)]
        return ", ".join(citations)
    
    def get_source_list(self) -> List[Dict]:
        unique_sources = list(set(self.sources))
        return [
            {
                'filename': src,
                'citation': parse_canlii_citation(src)
            }
            for src in sorted(unique_sources)
        ]


class RAGPipeline:
    
    def __init__(
        self,
        retriever: LegalCaseRetriever,
        llm_provider: Optional[LLMProvider] = None,
        max_context_tokens: int = 4000,
        min_relevance_score: float = 0.3
    ):

        self.retriever = retriever
        self.llm_provider = llm_provider
        self.max_context_tokens = max_context_tokens
        self.min_relevance_score = min_relevance_score
    
    def retrieve_and_filter(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve relevant chunks and filter by relevance threshold."""
        # Check if query mentions specific case citations (e.g., "2025 CanLII 125773")
        mentioned_case_files = extract_case_citations_from_query(query)
        
        # Check if query mentions case names (e.g., "R. v. Vrbanic")
        mentioned_case_names = extract_case_names_from_query(query)
        
        # If case names detected, search for files containing those names
        if mentioned_case_names:
            print(f"Detected case name: {', '.join(mentioned_case_names)}")
            # Retrieve more chunks to search through
            all_chunks = self.retriever.retrieve(query, top_k=top_k * 3)
            
            # Find which files contain the mentioned case names
            matching_files = set()
            for chunk in all_chunks:
                chunk_text = chunk['text']
                for case_name in mentioned_case_names:
                    # Check if case name appears in chunk text (case-insensitive)
                    if case_name.lower() in chunk_text.lower():
                        matching_files.add(chunk['source_file'])
                        break
            
            if matching_files:
                print(f"Found case in: {', '.join([parse_canlii_citation(f.replace('.txt', '')) for f in matching_files])}")
                mentioned_case_files.extend(list(matching_files))
            else:
                print(f"Warning: Could not locate '{', '.join(mentioned_case_names)}' in vector database")
        
        # If we have identified case files (either by citation or name)
        if mentioned_case_files:
            if not mentioned_case_names:  # Only print if not already printed
                print(f"Detected citation: {', '.join([parse_canlii_citation(c.replace('.txt', '')) for c in mentioned_case_files])}")
            
            print("Retrieving all chunks from mentioned case(s)...")
            
            # Get ALL chunks from the mentioned case files (not just top-k)
            priority_chunks = self.retriever.retrieve_by_source(mentioned_case_files)
            
            if priority_chunks:
                print(f"Retrieved {len(priority_chunks)} chunks from mentioned case(s)")
                
                # Get additional context from semantically similar chunks in other cases
                if len(priority_chunks) < top_k:
                    print(f"Retrieving {top_k - len(priority_chunks)} additional context chunks...")
                    other_chunks = self.retriever.retrieve(query, top_k=top_k)
                    # Filter out chunks from mentioned cases
                    other_chunks = [
                        chunk for chunk in other_chunks
                        if chunk['source_file'] not in mentioned_case_files
                    ]
                    # Combine: prioritize mentioned case chunks, then add other context
                    chunks = priority_chunks + other_chunks[:top_k - len(priority_chunks)]
                else:
                    # Use only chunks from mentioned case(s)
                    chunks = priority_chunks
            else:
                print(f"Warning: No chunks found from mentioned case(s), using general retrieval")
                chunks = self.retriever.retrieve(query, top_k=top_k)
        else:
            # Normal retrieval
            chunks = self.retriever.retrieve(query, top_k=top_k)
        
        # Filter by relevance threshold
        filtered_chunks = [
            chunk for chunk in chunks 
            if chunk['relevance_score'] >= self.min_relevance_score
        ]
        
        if not filtered_chunks:
            print(f"Warning: No chunks met relevance threshold of {self.min_relevance_score}")
            filtered_chunks = chunks[:3]
        
        return filtered_chunks
    
    def fit_context_budget(self, chunks: List[Dict]) -> List[Dict]:
        """Fit chunks within token budget, prioritizing highest relevance."""
        selected_chunks = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = chunk['token_count']
            
            if total_tokens + chunk_tokens > self.max_context_tokens:
                break
            
            selected_chunks.append(chunk)
            total_tokens += chunk_tokens
        
        if not selected_chunks:
            selected_chunks = [chunks[0]]
        
        return selected_chunks
    
    def build_context_string(self, chunks: List[Dict], include_metadata: bool = True) -> str:
        """Build formatted context string from chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            if include_metadata:
                citation = parse_canlii_citation(chunk['source_file'])
                header = (
                    f"[CONTEXT {i}]\n"
                    f"Case: {citation}\n"
                    f"Source File: {chunk['source_file']}\n"
                    f"Relevance: {chunk['relevance_score']:.3f}\n"
                    f"---"
                )
                context_parts.append(header)
            
            context_parts.append(chunk['text'])
            
            if include_metadata:
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def build_grounded_prompt(
        self,
        query: str,
        context: str,
        system_instructions: Optional[str] = None
    ) -> Dict[str, str]:
        """Build a grounded prompt with retrieved context."""
        if system_instructions is None:
            system_instructions = """
You are a legal assistant specializing in Canadian case law analysis.

Your role:
- Answer legal questions using ONLY the provided case excerpts.
- Do not rely on external knowledge.
- Do not fabricate case details, holdings, or citations.
- Clearly distinguish between what the case explicitly states and reasonable inferences drawn from the text.
- Acknowledge when the provided excerpts are insufficient to fully answer the question.

Guidelines:
- Base your answer strictly on the provided context.
- Quote or paraphrase relevant passages where appropriate.
- Cite cases using their proper CanLII citation format shown in the context headers.
- If the answer cannot be determined from the excerpts, state:
  "The answer cannot be determined from the provided case excerpts."
  """

        user_prompt = f"""**RELEVANT CASE LAW CONTEXT:**

{context}

---

**USER QUESTION:**
{query}

**INSTRUCTIONS:**
Answer the question using ONLY the case law context provided above. 
Cite cases using their proper CanLII citation format as shown in the context headers.
If the context doesn't contain sufficient information to answer completely, acknowledge this."""

        return {
            'system': system_instructions,
            'user': user_prompt
        }
    
    def process_query(
        self,
        query: str,
        top_k: int = 10,
        include_metadata: bool = True,
        custom_system_prompt: Optional[str] = None
    ) -> Dict:
        """Complete RAG pipeline: retrieve, filter, and build grounded prompt."""
        print(f"\nRetrieving relevant case law for: \"{query}\"")
        retrieved_chunks = self.retrieve_and_filter(query, top_k=top_k)
        
        selected_chunks = self.fit_context_budget(retrieved_chunks)
        total_tokens = sum(chunk['token_count'] for chunk in selected_chunks)
        
        context_string = self.build_context_string(selected_chunks, include_metadata)
        
        prompts = self.build_grounded_prompt(query, context_string, custom_system_prompt)
        
        sources = [chunk['source_file'] for chunk in selected_chunks]
        context = RAGContext(
            query=query,
            retrieved_chunks=retrieved_chunks,
            total_chunks=len(selected_chunks),
            sources=sources,
            total_tokens=total_tokens
        )
        
        print(f"Sources: {context.get_source_citations()}")
        
        return {
            'prompts': prompts,
            'context': context,
            'chunks': selected_chunks
        }
    
    def generate_answer(
        self,
        query: str,
        top_k: int = 10,
        temperature: float = 0.1
    ) -> Dict:
        if not self.llm_provider:
            raise ValueError("LLM provider not configured. Use process_query() for prompt generation only.")
        
        result = self.process_query(query, top_k=top_k)
        llm_result = self.llm_provider.generate_response(
            system_prompt=result['prompts']['system'],
            user_prompt=result['prompts']['user'],
            temperature=temperature
        )
        
        if llm_result.get('error'):
            print(f"Error: {llm_result['error']}")
            result['llm_response'] = None
            result['llm_error'] = llm_result['error']
        else:
            result['llm_response'] = llm_result['response']
            result['llm_tokens'] = llm_result['tokens_used']
            result['llm_model'] = llm_result['model']
        
        return result
    
    def get_pipeline_summary(self, result: Dict) -> str:
        """Generate human-readable summary of pipeline execution."""
        context = result['context']
        
        summary = f"""
RAG PIPELINE SUMMARY

Query: {context.query}

Context Retrieved:
  Total chunks selected: {context.total_chunks}
  Total tokens used: {context.total_tokens} / {self.max_context_tokens}
  Token utilization: {(context.total_tokens / self.max_context_tokens * 100):.1f}%
  
Sources Referenced (CanLII Citations):
  {context.get_source_citations()}
  
Relevance Scores:"""
        
        for chunk in result['chunks']:
            citation = parse_canlii_citation(chunk['source_file'])
            summary += f"\n  {citation} (chunk {chunk['chunk_id']}): {chunk['relevance_score']:.3f}"
        
        summary += "\n"
        
        return summary


def main():
    from dotenv import load_dotenv
    
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        return
    
    db_path = project_root / "app" / "data" / "vector-db"
    retriever = LegalCaseRetriever(str(db_path), api_key.strip().strip("\"'\'"))
    llm_provider = LLMProvider(api_key.strip().strip("\"'\'"), model="gpt-4o-mini")
    
    rag = RAGPipeline(
        retriever=retriever,
        llm_provider=llm_provider,
        max_context_tokens=4000,
        min_relevance_score=0.3
    )
    
    print("\nLegal AI")
    print("Ask questions about Canadian case law.")
    print("LegalAI is for research purposes only. NOT legal advice. Some recent cases may be incomplete.")
    print("Commands: 'quit' or 'exit' to stop, 'summary' to see details\n")
    
    last_result = None
    
    while True:
        try:
            query = input("\nYour question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if query.lower() == 'summary' and last_result:
                print(rag.get_pipeline_summary(last_result))
                if 'llm_tokens' in last_result:
                    print(f"\nLLM Tokens Used:")
                    print(f"  Prompt: {last_result['llm_tokens']['prompt']}")
                    print(f"  Completion: {last_result['llm_tokens']['completion']}")
                    print(f"  Total: {last_result['llm_tokens']['total']}")
                continue
            elif query.lower() == 'summary':
                print("No previous query to summarize.")
                continue
            
            result = rag.generate_answer(query, top_k=8, temperature=0.1)
            last_result = result
            
            if result.get('llm_response'):
                print("\nANSWER:")
                print(f"\n{result['llm_response']}\n")
                
                # CanLII citations
                sources_list = result['context'].get_source_list()
                if len(sources_list) == 1:
                    print(f"Source: {sources_list[0]['citation']}")
                else:
                    print(f"Sources ({len(sources_list)} cases):")
                    for src in sources_list:
                        print(f"  - {src['citation']}")
                
                print(f"Context: {result['context'].total_chunks} chunks, {result['context'].total_tokens} tokens")
                print(f"Model: {result['llm_model']} | Tokens: {result['llm_tokens']['total']}")
            else:
                print(f"\nFailed to generate answer: {result.get('llm_error', 'Unknown error')}")
            
        except KeyboardInterrupt:
            print("\n\nUser interrupted. Exiting.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

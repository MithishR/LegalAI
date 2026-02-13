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

from retrieval.retriever import LegalCaseRetriever
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
    """Extract case names from query, handling both criminal (R. v.) and civil (X v. Y) cases."""
    case_names = []    
    pattern1 = r'R\.?\s*v\.?\s+([A-Z][a-zA-Z]+(?:\s+(?:and|et)\s+[A-Z][a-zA-Z]+)?)'
    matches1 = re.findall(pattern1, query)
    for name in matches1:
        case_names.append(f"R. v. {name}")    
    pattern2 = r'([A-Z][a-zA-Z]+(?:\s+[a-zA-ZÀ-ÿ]+){1,10})\s+v[s]?\.?\s+([A-Z][a-zA-Z]+(?:\s+[a-zA-ZÀ-ÿ]+){0,10})'
    matches2 = re.findall(pattern2, query, re.IGNORECASE)
    for party1, party2 in matches2:
        if not party1.strip().upper().startswith('R'):
            case_names.append(f"{party1.strip()} v. {party2.strip()}")
    
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
        self.conversation_history = []  # List of {'question': str, 'answer': str}
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")
    
    def retrieve_and_filter(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve relevant chunks and filter by relevance threshold."""
        mentioned_case_files = extract_case_citations_from_query(query)
        mentioned_case_names = extract_case_names_from_query(query)
        if mentioned_case_names:
            print(f"Detected case name: {', '.join(mentioned_case_names)}")
            all_chunks = self.retriever.retrieve(query, top_k=top_k * 3)
            matching_files = set()
            for chunk in all_chunks:
                chunk_text = chunk['text']
                normalized_chunk = ' '.join(chunk_text.lower().split())
                for case_name in mentioned_case_names:
                    normalized_case_name = ' '.join(case_name.lower().split())
                    if normalized_case_name in normalized_chunk:
                        matching_files.add(chunk['source_file'])
                        break
            
            if matching_files:
                print(f"Found case in: {', '.join([parse_canlii_citation(f.replace('.txt', '')) for f in matching_files])}")
                mentioned_case_files.extend(list(matching_files))
            else:
                print(f"Warning: Could not locate '{', '.join(mentioned_case_names)}' in vector database")
        
        if mentioned_case_files:
            if not mentioned_case_names:  
                print(f"Detected citation: {', '.join([parse_canlii_citation(c.replace('.txt', '')) for c in mentioned_case_files])}")
            
            print("Retrieving all chunks from mentioned case(s)...")            
            priority_chunks = self.retriever.retrieve_by_source(mentioned_case_files)
            
            if priority_chunks:
                print(f"Retrieved {len(priority_chunks)} chunks from mentioned case(s)")                
                header_chunks = [c for c in priority_chunks if c['chunk_id'] <= 2]
                other_chunks = [c for c in priority_chunks if c['chunk_id'] > 2]
                priority_chunks = header_chunks + other_chunks
                if len(priority_chunks) < top_k:
                    print(f"Retrieving {top_k - len(priority_chunks)} additional context chunks...")
                    other_chunks = self.retriever.retrieve(query, top_k=top_k)
                    other_chunks = [
                        chunk for chunk in other_chunks
                        if chunk['source_file'] not in mentioned_case_files
                    ]
                    chunks = priority_chunks + other_chunks[:top_k - len(priority_chunks)]
                else:
                    chunks = priority_chunks
            else:
                print(f"Warning: No chunks found from mentioned case(s), using general retrieval")
                chunks = self.retriever.retrieve(query, top_k=top_k)
        else:
            chunks = self.retriever.retrieve(query, top_k=top_k)        
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
        """Build a grounded prompt with retrieved context and conversation history."""
        if system_instructions is None:
            system_instructions = """
You are a legal assistant specializing in Canadian case law analysis.

Your role:
- Answer ALL types of questions about cases using ONLY the provided case excerpts.
- Handle both FACTUAL questions (parties, judges, facts, dates, procedural history, outcomes) and LEGAL ANALYSIS questions (holdings, reasoning, legal tests, precedents).
- Maintain conversation context - reference previous questions and answers when relevant.
- Do not rely on external knowledge.
- Do not fabricate any information.

For factual questions (who, what, when, where, outcome):
- Extract specific information: party names, judges, dates, facts, procedural history, and decisions.
- Look for party names in case headers (Appellant, Respondent, Plaintiff, Defendant, Crown, etc.).
- Quote exact names and details from the case text.
- State what the case was about in clear, factual terms.

For legal analysis questions (why, how, legal test, standard):
- Explain the court's reasoning and legal principles.
- Identify relevant legal tests, standards, or frameworks applied.
- Distinguish between holdings, dicta, and analysis.
- Quote or paraphrase relevant passages.

For follow-up questions:
- Reference information from the conversation history when appropriate.
- Use pronouns like "that case" or "the defendant" when context is clear.
- Maintain continuity across the conversation.

Guidelines:
- Base your answer strictly on the provided context.
- Cite cases using their proper CanLII citation format shown in the context headers.
- If specific information cannot be found in the excerpts, state clearly:
  "This information is not available in the provided case excerpts."
- Be precise, direct, and avoid speculation.
  """
        conversation_context = ""
        if self.conversation_history:
            conversation_context = "**CONVERSATION HISTORY:**\n"
            for i, turn in enumerate(self.conversation_history[-5:], 1):  # Last 5 turns
                conversation_context += f"\nQ{i}: {turn['question']}\n"
                conversation_context += f"A{i}: {turn['answer']}\n"
            conversation_context += "\n---\n\n"

        user_prompt = f"""{conversation_context}**RELEVANT CASE LAW CONTEXT:**

{context}

---

**CURRENT QUESTION:**
{query}

**INSTRUCTIONS:**
Answer the current question using the case law context provided above.
If this is a follow-up question, reference the conversation history as needed.
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
            self.conversation_history.append({
                'question': query,
                'answer': result['llm_response']
            })
        
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
    print("Commands: 'quit' or 'exit' to stop, 'summary' to see details, 'clear' to reset conversation\n")
    
    last_result = None
    
    while True:
        try:
            query = input("\nYour question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if query.lower() in ['clear', 'reset']:
                rag.clear_conversation()
                continue
            
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
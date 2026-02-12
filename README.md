# LegalAI
LegalAI is a retrieval-augmented generation (RAG) system for Canadian case law. It allows users to ask questions about Supreme Court and lower court decisions and receive grounded answers based on uploaded case law documents.

The system uses:
- ChromaDB for storing vector embeddings of case law chunks
- OpenAI embeddings and LLM for contextual retrieval and answer generation
- Text chunking to split cases into 1000-token segments

## Using the RAG Pipeline

- Run the interactive agent:
``` python app/agent/rag_pipeline.py```
- Commands:
1. Ask questions about Canadian case law.
2. Type quit or exit to stop.

Example queries:
```What is the standard for 'grossly disproportionate' sentences?```
Output: 
```
The standard for 'grossly disproportionate' sentences is described as being "so excessive as to outrage standards of decency" and "abhorrent or intolerable" to society (R. v. Lloyd, 2016 SCC 13, [2016] S.C.R. 130, at para. 24).

The standard is intended to be demanding and will be attained only rarely (R. v. Bissonnette, 2022 SCC 23, at para. 70). To establish that a sentence is grossly disproportionate, it must be shown that the minimum sentence goes beyond what is necessary to achieve Parliament’s objectives relevant to the offence (2023 SCC 26, para. 226). 

Additionally, the analysis under section 12 of the Charter involves determining whether the mandatory punishment is grossly disproportionate when compared to the fit sentence for either the claimant or for a reasonable hypothetical offender (R. v. Boudreault, 2018 SCC 58, [2018] 3 S.C.R. 599, at para. 46). This requires a contextual comparison between the fit sentence and the impugned mandatory minimum (2023 SCC 26, para. 182).
```
Note: Some recent cases (e.g., R. v. Vrbanic, 2025 CanLII 125773) may only have a summary or header in the database. Full reasoning must be added for complete answers.

**Legal and Compliance Notes**
- All cases are sourced from CanLII, which allows reproduction for research and legal analysis, with proper attribution.
- Bulk downloads or systematic scraping are prohibited. Always comply with CanLII Terms of Use.
- LegalAI does not provide legal advice; it only summarizes case law contextually.

**Limitations**
- Incomplete case content: Some judgments (e.g., very recent ones) may not be fully available in the raw text files; answers may be partial.
- Not legal advice: Outputs summarize case law contextually; users must not rely on LegalAI for legal decisions. I am not a lawyer and I cannot verify the legal correctness of the content.
- Context window: Only a limited number of tokens (default 4000) can be used for grounding; very long cases may be truncated.
- Potential errors: Embedding-based retrieval may occasionally retrieve less relevant chunks, leading to incomplete or ambiguous answers.
- CanLII compliance: Bulk downloading or scraping beyond individual manual downloads violates CanLII terms; ensure proper attribution and lawful use.

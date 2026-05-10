import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
APP_ROOT     = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from retrieval.retriever import LegalCaseRetriever
from models.llm_provider import LLMProvider
from agent.rag_pipeline import RAGPipeline

RESULTS_DIR      = Path(__file__).parent / "results"
DEFAULT_QA_FILE  = RESULTS_DIR / "all_qa_pairs.json"
DB_PATH          = APP_ROOT / "data" / "vector-db"
TOP_K            = 10
def check_source_match(expected_source: str, retrieved_chunks: List[Dict]) -> tuple[bool, int | None]:
    for i, chunk in enumerate(retrieved_chunks):
        if chunk["source_file"] == expected_source:
            return True, i + 1
    return False, None


def check_keyword_match(answer_text: str, keywords: List[str]) -> tuple[bool, List[str], List[str]]:
    answer_lower = answer_text.lower()
    matched  = [kw for kw in keywords if kw.lower() in answer_lower]
    missing  = [kw for kw in keywords if kw.lower() not in answer_lower]
    return len(missing) == 0, matched, missing
def run_evaluation(
    qa_pairs: List[Dict],
    rag: RAGPipeline,
) -> Dict:
    results = []
    source_correct  = 0
    keyword_correct = 0
    total           = len(qa_pairs)

    for i, item in enumerate(qa_pairs, 1):
        question        = item["question"]
        expected_source = item.get("source_file", "")
        keywords        = item.get("keywords", [])

        print(f"\n[{i:03d}/{total}] {question[:80]}...")

        chunks = rag.retriever.retrieve(
            query=question,
            top_k=TOP_K,
            method=rag.retrieval_method,
            alpha=rag.hybrid_alpha,
        )

        source_matched, rank = check_source_match(expected_source, chunks)
        if source_matched:
            source_correct += 1
            print(f"  Source : PASS  (rank {rank})  {expected_source}")
        else:
            retrieved_top3 = [c["source_file"] for c in chunks[:3]]
            print(f"  Source : FAIL  expected={expected_source}  got={retrieved_top3}")

        llm_answer      = None
        kw_matched_flag = None
        kw_matched      = []
        kw_missing      = []

        result = rag.generate_answer(question, top_k=TOP_K, temperature=0.1)
        llm_answer = result.get("llm_response") or ""

        if keywords and llm_answer:
            kw_matched_flag, kw_matched, kw_missing = check_keyword_match(llm_answer, keywords)
            if kw_matched_flag:
                keyword_correct += 1
                print(f"  Keywords: PASS  {kw_matched}")
            else:
                print(f"  Keywords: FAIL  matched={kw_matched}  missing={kw_missing}")
        elif not keywords:
            print("  Keywords: SKIP (no keywords defined)")

        rag.clear_conversation()

        results.append({
            "id":              i,
            "question":        question,
            "expected_source": expected_source,
            "expected_answer": item.get("answer", ""),
            "keywords":        keywords,
            "source_match":    source_matched,
            "source_rank":     rank,
            "llm_answer":      llm_answer,
            "keyword_match":   kw_matched_flag,
            "keywords_found":  kw_matched,
            "keywords_missing": kw_missing,
        })

    kw_eligible = sum(1 for r in results if r["keyword_match"] is not None)

    report = {
        "timestamp":              datetime.utcnow().isoformat() + "Z",
        "total_questions":        total,
        "top_k":                  TOP_K,
        "source_correct":         source_correct,
        "source_accuracy_pct":    round(source_correct / total * 100, 2),
        "keyword_eligible":       kw_eligible,
        "keyword_correct":        keyword_correct,
        "keyword_accuracy_pct":   round(keyword_correct / kw_eligible * 100, 2) if kw_eligible else None,
        "per_question":           results,
    }

    return report

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    if not DEFAULT_QA_FILE.exists():
        print(f"ERROR: Q/A file not found: {DEFAULT_QA_FILE}")
        print("Run: python app/evaluation/generate_qa_pairs.py --all")
        sys.exit(1)

    with open(DEFAULT_QA_FILE) as f:
        qa_pairs = json.load(f)

    print(f"Loaded {len(qa_pairs)} Q/A pair(s) from {DEFAULT_QA_FILE.name}")

    print(f"Loading vector database from: {DB_PATH}")
    retriever = LegalCaseRetriever(
        db_path=str(DB_PATH),
        api_key=api_key,
        enable_bm25=True,
    )

    rag = RAGPipeline(
        retriever=retriever,
        llm_provider=LLMProvider(api_key=api_key, model="gpt-4o-mini"),
        max_context_tokens=4000,
        min_relevance_score=0.3,
        retrieval_method="hybrid",
        hybrid_alpha=0.5,
    )

    print(f"Questions: {len(qa_pairs)}\n")

    report = run_evaluation(qa_pairs, rag)
    print(f"  Total questions   : {report['total_questions']}")
    print(f"  Retrieval Accuracy (Source Match)")
    print(f"    Correct         : {report['source_correct']} / {report['total_questions']}")
    print(f"    Accuracy        : {report['source_accuracy_pct']:.1f}%")
    print(f"  Answer Fidelity (Keyword Match)")
    if report["keyword_eligible"]:
        print(f"    Eligible        : {report['keyword_eligible']} questions")
        print(f"    Correct         : {report['keyword_correct']} / {report['keyword_eligible']}")
        print(f"    Accuracy        : {report['keyword_accuracy_pct']:.1f}%")
    else:
        print("    No keyword data available.")

    failed_source = [r for r in report["per_question"] if not r["source_match"]]
    if failed_source:
        print(f"\n  Failed source retrievals ({len(failed_source)}):")
        for r in failed_source:
            print(f"    Q{r['id']:03d}: {r['question'][:60]}...")
            print(f"         expected: {r['expected_source']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"rag_eval_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to: {out_path}")


if __name__ == "__main__":
    main()

"""
Q/A Pair Generator
Takes a chunked-case JSON file (or a single chunk) and a prompt template,
and uses the LLM to generate verified Question/Answer pairs for the benchmark.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
APP_ROOT     = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from models.llm_provider import LLMProvider

CHUNKS_DIR   = APP_ROOT / "data" / "chunked-cases"
RESULTS_DIR  = Path(__file__).parent / "results"
DEFAULT_PROMPT = """\
You are a legal research assistant helping to build a benchmark evaluation set \
for a Supreme Court of Canada retrieval system.

Given the following excerpt from a Canadian court case, generate exactly \
{n_pairs} factual Question/Answer pairs that:
- Have a single, unambiguous correct answer
- Can be answered using only information present in this excerpt
- Cover different facts (parties, outcome, judges, legal issue, date, etc.)
- Are phrased neutrally — do not mention "the excerpt" or "the text"

Return ONLY a valid JSON array.  No markdown fences, no extra keys.
Each object must have exactly these keys:
  "question" : the question string
  "answer"   : the correct answer string
  "keywords" : a list of 2-5 key words/phrases that must appear in a correct answer

Excerpt:
{chunk}
"""

SYSTEM_PROMPT = (
    "You are a precise legal assistant. Output only valid JSON. "
    "Never add commentary outside the JSON array."
)


def generate_pairs(
    chunk_text: str,
    llm: LLMProvider,
    prompt_template: str,
    n_pairs: int = 3,
) -> List[Dict]:
    user_prompt = prompt_template.format(chunk=chunk_text, n_pairs=n_pairs)
    result = llm.generate_response(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=1000,
    )

    if result.get("error"):
        print(f"  [LLM ERROR] {result['error']}")
        return []

    raw = result["response"].strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        pairs = json.loads(raw)
        if not isinstance(pairs, list):
            print("  [PARSE ERROR] LLM returned non-list JSON")
            return []
        return pairs
    except json.JSONDecodeError as e:
        print(f"  [PARSE ERROR] {e}")
        print(f"  Raw output:\n{raw[:300]}")
        return []


def process_chunk_file(
    chunk_path: Path,
    llm: LLMProvider,
    prompt_template: str,
    n_pairs: int,
    chunk_index: int = 0,
) -> List[Dict]:
    with open(chunk_path) as f:
        chunks = json.load(f)

    if not chunks:
        print(f"  [SKIP] No chunks in {chunk_path.name}")
        return []

    chunk = chunks[chunk_index]
    source_file = chunk.get("source_file", chunk_path.stem.replace("_chunks", ".txt"))
    print(f"  chunk_id={chunk['chunk_id']}  tokens={chunk.get('token_count', '?')}")

    pairs = generate_pairs(chunk["text"], llm, prompt_template, n_pairs)

    for pair in pairs:
        pair["source_file"] = source_file
        pair["chunk_id"]    = chunk["chunk_id"]

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate Q/A pairs from legal case chunks using an LLM"
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source", metavar="FILE",
        help="Filename of a chunked-case JSON in data/chunked-cases/ (e.g. 2024scc40_chunks.json)"
    )
    source_group.add_argument(
        "--text", metavar="TEXT",
        help="Raw text chunk to generate Q/A pairs from"
    )
    source_group.add_argument(
        "--all", action="store_true",
        help="Process every chunk file in data/chunked-cases/"
    )

    parser.add_argument(
        "--chunk-index", type=int, default=0, metavar="N",
        help="Which chunk (0-indexed) to use from the file (default: 0 = case header)"
    )
    parser.add_argument(
        "--n-pairs", type=int, default=3, metavar="N",
        help="Number of Q/A pairs to generate per chunk (default: 3)"
    )
    parser.add_argument(
        "--prompt", metavar="TEMPLATE",
        help="Custom prompt template string.  Use {chunk} and {n_pairs} as placeholders."
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Write results to this JSON file (default: results/qa_pairs_<source>.json)"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    llm = LLMProvider(api_key=api_key, model=args.model)
    prompt_template = args.prompt or DEFAULT_PROMPT

    all_pairs: List[Dict] = []

    if args.text:
        print("Generating Q/A pairs from provided text...")
        pairs = generate_pairs(args.text, llm, prompt_template, args.n_pairs)
        for pair in pairs:
            pair["source_file"] = "manual_input"
            pair["chunk_id"]    = 0
        all_pairs.extend(pairs)

    elif args.source:
        chunk_path = CHUNKS_DIR / args.source
        if not chunk_path.exists():
            print(f"ERROR: File not found: {chunk_path}")
            sys.exit(1)
        print(f"Processing {args.source}  (chunk index {args.chunk_index})...")
        all_pairs.extend(
            process_chunk_file(chunk_path, llm, prompt_template, args.n_pairs, args.chunk_index)
        )

    else:  # --all
        chunk_files = sorted(CHUNKS_DIR.glob("*_chunks.json"))
        print(f"Processing {len(chunk_files)} chunk files...")
        for idx, chunk_path in enumerate(chunk_files, 1):
            print(f"[{idx:02d}/{len(chunk_files)}] {chunk_path.name}")
            pairs = process_chunk_file(chunk_path, llm, prompt_template, args.n_pairs, args.chunk_index)
            all_pairs.extend(pairs)

    print(f"\nGenerated {len(all_pairs)} Q/A pair(s):\n")
    for i, pair in enumerate(all_pairs, 1):
        print(f"  [{i}] Q: {pair.get('question', '')}")
        print(f"       A: {pair.get('answer', '')}")
        kw = pair.get("keywords", [])
        if kw:
            print(f"       Keywords: {', '.join(kw)}")
        print()

    if all_pairs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if args.output:
            out_path = Path(args.output)
        else:
            if args.text:
                stem = "manual_input"
            elif args.source:
                stem = Path(args.source).stem
            else:
                stem = "all"
            out_path = RESULTS_DIR / f"qa_pairs_{stem}.json"

        with open(out_path, "w") as f:
            json.dump(all_pairs, f, indent=2)
        print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()

"""
Fast parallel taxonomy label extraction via OpenRouter.

Uses concurrent.futures to send 20 parallel API requests instead of sequential.
59K sentences should take ~30 min instead of ~9 hours.

Usage:
    OPENROUTER_API_KEY=... python3 src/data_pipeline/extract_taxonomy_fast.py \
        --corpus data/cot_corpus_diverse/corpus.jsonl \
        --output data/cot_corpus_diverse/labels_taxonomy.jsonl \
        --workers 20
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

TAXONOMY_PROMPT = """Classify this chain-of-thought reasoning sentence into exactly ONE category.

Categories:
- problem_setup: Restating or parsing the problem
- plan_generation: Planning next steps or strategy
- fact_retrieval: Recalling a fact, formula, or definition
- active_computation: Performing calculation or logical deduction
- uncertainty_management: Expressing doubt, considering alternatives
- result_consolidation: Combining intermediate results
- self_checking: Verifying or double-checking work
- final_answer: Stating the final answer

Context (preceding sentences): {context}
Sentence to classify: {sentence}

Respond with ONLY the category name, nothing else."""

VALID_CATEGORIES = [
    "problem_setup", "plan_generation", "fact_retrieval",
    "active_computation", "uncertainty_management",
    "result_consolidation", "self_checking", "final_answer",
]


def classify_one(item, api_key):
    """Classify a single sentence. Returns the label dict."""
    prompt = TAXONOMY_PROMPT.format(context=item["context"], sentence=item["sentence_text"])
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "google/gemini-2.0-flash-001",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.0,
            },
            timeout=30,
        )
        result = response.json()
        category = result["choices"][0]["message"]["content"].strip().lower()

        if category not in VALID_CATEGORIES:
            for vc in VALID_CATEGORIES:
                if vc in category:
                    category = vc
                    break
            else:
                category = "active_computation"

        return {
            "id": item["id"],
            "sentence_idx": item["sentence_idx"],
            "sentence_text": item["sentence_text"],
            "category": category,
        }
    except Exception as e:
        return {
            "id": item["id"],
            "sentence_idx": item["sentence_idx"],
            "sentence_text": item["sentence_text"],
            "category": "active_computation",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    api_key = __import__("os").environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    # Load corpus
    corpus = []
    with open(args.corpus) as f:
        for line in f:
            corpus.append(json.loads(line))
    print(f"Loaded {len(corpus)} corpus entries")

    # Build batch of all sentences
    batch = []
    for entry in corpus:
        sentences = entry["sentences"]
        for s_idx, sentence in enumerate(sentences):
            context = " ".join(sentences[max(0, s_idx - 3):s_idx])
            batch.append({
                "id": entry["id"],
                "sentence_idx": s_idx,
                "sentence_text": sentence,
                "context": context,
            })

    print(f"Classifying {len(batch)} sentences with {args.workers} workers...")

    # Process in parallel
    labels = [None] * len(batch)
    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(classify_one, item, api_key): idx
            for idx, item in enumerate(batch)
        }
        with tqdm(total=len(batch), desc="Taxonomy") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()
                labels[idx] = result
                if "error" in result:
                    errors += 1
                pbar.update(1)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")

    print(f"\nWrote {len(labels)} labels to {args.output}")
    if errors:
        print(f"  ({errors} had API errors, defaulted to active_computation)")

    # Print category distribution
    from collections import Counter
    dist = Counter(l["category"] for l in labels)
    print("\nCategory distribution:")
    for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({100*count/len(labels):.1f}%)")


if __name__ == "__main__":
    main()

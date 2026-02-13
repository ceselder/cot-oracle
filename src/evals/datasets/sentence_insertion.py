"""
Eval: Sentence Insertion Detection (needle-in-a-haystack)

Insert a sentence from a *different* problem's CoT into the middle of
the current CoT. Oracle must identify which sentence was inserted.

This is the first eval requiring the oracle to read an *entire trajectory*
(activations at every sentence boundary), directly testing trajectory-
reading capability needed for the full delta-sequence architecture.

50/50 split: half have an inserted sentence, half are clean (answer "none").

Requires the CoT corpus to exist (data/cot_corpus/corpus.jsonl).
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem


def generate_sentence_insertion_dataset(
    n: int = 100,
    seed: int = 42,
    corpus_path: str = "data/cot_corpus/corpus.jsonl",
) -> list[EvalItem]:
    """Generate sentence insertion detection eval examples.

    50% of items: a foreign sentence is inserted (oracle should identify it).
    50% of items: clean CoT, no insertion (oracle should say "none").

    Each item's test_prompt is the (possibly spliced) CoT text.
    The oracle sees activations at ALL sentence boundaries.
    Ground truth is stored in metadata["inserted_step"] (1-indexed, or "none").
    """
    random.seed(seed)

    # Load corpus
    corpus = []
    corpus_path = Path(corpus_path)
    if corpus_path.exists():
        with open(corpus_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if len(entry.get("sentences", [])) >= 5:
                        corpus.append(entry)

    if len(corpus) < 10:
        # Fallback: generate minimal items from hardcoded data
        return _generate_minimal(n, seed)

    items = []

    for i in range(n):
        is_insertion = i % 2 == 0  # 50/50 split

        # Pick a host CoT
        host = random.choice(corpus)
        host_sentences = list(host["sentences"])

        if is_insertion:
            # Pick a donor sentence from a DIFFERENT problem
            donor = random.choice([c for c in corpus if c["id"] != host["id"]])
            donor_sentence = random.choice(donor["sentences"])

            # Insert at a random position (not first or last)
            insert_pos = random.randint(1, len(host_sentences) - 1)
            spliced = host_sentences[:insert_pos] + [donor_sentence] + host_sentences[insert_pos:]
            inserted_step = insert_pos + 1  # 1-indexed

            # Build the spliced CoT text
            spliced_text = " ".join(spliced)

            # The question asked to the oracle
            oracle_target = str(inserted_step)
        else:
            spliced = host_sentences
            spliced_text = " ".join(spliced)
            inserted_step = "none"
            oracle_target = "none"

        # The test_prompt contains the question + spliced CoT for context
        # The clean_prompt is just the question (for extracting clean answer)
        question = host["question"]

        items.append(EvalItem(
            eval_name="sentence_insertion",
            example_id=f"insertion_{i:04d}",
            clean_prompt=question,
            test_prompt=question,  # Same question; the spliced CoT is in metadata
            correct_answer=host.get("correct_answer", ""),
            nudge_answer=None,
            metadata={
                "host_id": host["id"],
                "n_sentences": len(spliced),
                "original_n_sentences": len(host_sentences),
                "inserted_step": inserted_step,
                "oracle_target": oracle_target,
                "spliced_cot_text": spliced_text,
                "spliced_sentences": spliced,
                "is_insertion": is_insertion,
                "donor_id": donor["id"] if is_insertion else None,
                "donor_sentence": donor_sentence if is_insertion else None,
            },
        ))

    return items


def _generate_minimal(n: int, seed: int) -> list[EvalItem]:
    """Fallback when corpus isn't available. Returns empty list with warning."""
    print("  WARNING: No corpus found for sentence insertion eval. Skipping.")
    return []

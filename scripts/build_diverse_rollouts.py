#!/usr/bin/env python3
"""
Download 12 HuggingFace datasets and prepare prompts for Qwen3-8B CoT rollout generation.

Also includes entries from existing local corpora (cot_corpus_v5, concept_corpus).

Output: data/diverse_rollouts/prompts.jsonl
"""

import json
import os
import random
import sys
import traceback
from pathlib import Path

# Reproducibility
SEED = 42
random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "diverse_rollouts"
OUTPUT_FILE = OUTPUT_DIR / "prompts.jsonl"


def format_choices(labels, texts):
    """Format multiple-choice options as A) ... B) ... etc."""
    parts = []
    for label, text in zip(labels, texts):
        parts.append(f"{label}) {text}")
    return "\n".join(parts)


def format_choices_from_list(texts, labels=None):
    """Format choices from a list of texts, auto-generating A/B/C/... labels."""
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(texts))]
    return format_choices(labels, texts)


def write_prompts(prompts, output_file):
    """Write prompts to JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def sample_items(items, target_count):
    """Sample target_count items from list, or return all if fewer."""
    if len(items) <= target_count:
        return items
    return random.sample(items, target_count)


# ─── Dataset loaders ────────────────────────────────────────────────────────────

def load_lmsys(target=20000):
    """lmsys/lmsys-chat-1m: diverse user prompts."""
    print(f"  Loading lmsys/lmsys-chat-1m (target={target})...")
    from datasets import load_dataset

    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)

    prompts = []
    seen = set()
    n_scanned = 0
    for item in ds:
        n_scanned += 1
        if len(prompts) >= target:
            break
        if n_scanned > 500000:  # safety cap on streaming iteration
            print(f"    Scanned {n_scanned} items, stopping")
            break
        # Filter: English, first turn, non-empty
        if item.get("language") != "English":
            continue
        conv = item.get("conversation", [])
        if not conv or len(conv) == 0:
            continue
        first_msg = conv[0]
        if first_msg.get("role") not in ("human", "user"):
            continue
        question = first_msg.get("content", "").strip()
        if not question or len(question) < 10:
            continue
        # Deduplicate
        key = question[:200]
        if key in seen:
            continue
        seen.add(key)
        prompts.append({
            "id": f"lmsys_{len(prompts):05d}",
            "source": "lmsys",
            "question": question,
            "metadata": {
                "conversation_id": item.get("conversation_id", ""),
                "model": item.get("model", ""),
            },
        })

    prompts = sample_items(prompts, target)
    print(f"    Collected {len(prompts)} prompts")
    return prompts


def load_aqua_rat(target=10000):
    """deepmind/aqua_rat: math word problems."""
    print(f"  Loading deepmind/aqua_rat (target={target})...")
    from datasets import load_dataset

    ds = load_dataset("deepmind/aqua_rat", split="train")

    prompts = []
    for i, item in enumerate(ds):
        question = item.get("question", "").strip()
        if not question:
            continue
        # Format options
        options = item.get("options", [])
        if options:
            question += "\n\nOptions:\n" + "\n".join(options)
        prompts.append({
            "id": f"aqua_rat_{i:05d}",
            "source": "aqua_rat",
            "question": question,
            "metadata": {
                "correct": item.get("correct", ""),
                "rationale": item.get("rationale", ""),
            },
        })

    prompts = sample_items(prompts, target)
    print(f"    Collected {len(prompts)} prompts")
    return prompts


def load_mmlu_pro(target=4800):
    """TIGER-Lab/MMLU-Pro: multi-choice advanced questions."""
    print(f"  Loading TIGER-Lab/MMLU-Pro (target={target})...")
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    prompts = []
    for i, item in enumerate(ds):
        question = item.get("question", "").strip()
        if not question:
            continue
        options = item.get("options", [])
        if options:
            formatted = format_choices_from_list(options)
            question = f"Q: {question}\n\nOptions:\n{formatted}"
        prompts.append({
            "id": f"mmlu_pro_{i:05d}",
            "source": "mmlu_pro",
            "question": question,
            "metadata": {
                "answer": item.get("answer", ""),
                "answer_index": item.get("answer_index", ""),
                "category": item.get("category", ""),
                "src": item.get("src", ""),
            },
        })

    prompts = sample_items(prompts, target)
    print(f"    Collected {len(prompts)} prompts")
    return prompts


def load_commonsense_qa(target=4000):
    """tau/commonsense_qa: commonsense reasoning multi-choice."""
    print(f"  Loading tau/commonsense_qa (target={target})...")
    from datasets import load_dataset

    ds = load_dataset("tau/commonsense_qa", split="train")

    prompts = []
    for i, item in enumerate(ds):
        question = item.get("question", "").strip()
        if not question:
            continue
        choices = item.get("choices", {})
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if labels and texts:
            formatted = format_choices(labels, texts)
            question = f"{question}\n\nOptions:\n{formatted}"
        prompts.append({
            "id": f"commonsense_qa_{i:05d}",
            "source": "commonsense_qa",
            "question": question,
            "metadata": {
                "answer": item.get("answerKey", ""),
                "concept": item.get("question_concept", ""),
            },
        })

    prompts = sample_items(prompts, target)
    print(f"    Collected {len(prompts)} prompts")
    return prompts


def load_gsm8k(target=3500):
    """openai/gsm8k: grade school math."""
    print(f"  Loading openai/gsm8k (target={target})...")
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")

    prompts = []
    for i, item in enumerate(ds):
        question = item.get("question", "").strip()
        if not question:
            continue
        answer_text = item.get("answer", "")
        # Extract final numeric answer after ####
        final_answer = ""
        if "####" in answer_text:
            final_answer = answer_text.split("####")[-1].strip()
        prompts.append({
            "id": f"gsm8k_{i:05d}",
            "source": "gsm8k",
            "question": question,
            "metadata": {
                "answer": final_answer,
                "solution": answer_text,
            },
        })

    prompts = sample_items(prompts, target)
    print(f"    Collected {len(prompts)} prompts")
    return prompts


def load_math(target=3200):
    """hendrycks/competition_math or lighteval/MATH: competition math."""
    print(f"  Loading competition_math / MATH (target={target})...")
    from datasets import load_dataset

    # EleutherAI/hendrycks_math has per-subject configs
    MATH_CONFIGS = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]

    ds = None
    # First try single-config datasets
    for name in ["hendrycks/competition_math", "lighteval/MATH",
                  "dim/competition_math", "qwedsacf/competition_math"]:
        try:
            ds = load_dataset(name, split="train")
            print(f"    Loaded from {name}")
            break
        except Exception as e:
            print(f"    Failed to load {name}: {e}")

    # If none worked, try EleutherAI with concatenated configs
    if ds is None:
        try:
            from datasets import concatenate_datasets
            parts = []
            for cfg in MATH_CONFIGS:
                part = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
                parts.append(part)
                print(f"    Loaded EleutherAI/hendrycks_math/{cfg}: {len(part)} items")
            ds = concatenate_datasets(parts)
            print(f"    Combined EleutherAI/hendrycks_math: {len(ds)} total items")
        except Exception as e:
            print(f"    Failed to load EleutherAI/hendrycks_math: {e}")

    if ds is None:
        print("    WARNING: Could not load any MATH dataset, skipping")
        return []

    prompts = []
    for i, item in enumerate(ds):
        question = item.get("problem", item.get("question", "")).strip()
        if not question:
            continue
        prompts.append({
            "id": f"math_{i:05d}",
            "source": "math",
            "question": question,
            "metadata": {
                "answer": item.get("solution", item.get("answer", "")),
                "level": item.get("level", ""),
                "type": item.get("type", item.get("subject", "")),
            },
        })

    prompts = sample_items(prompts, target)
    print(f"    Collected {len(prompts)} prompts")
    return prompts


def load_science_qa(target=1700):
    """derek-thomas/ScienceQA: science questions (text-only)."""
    print(f"  Loading derek-thomas/ScienceQA (target={target})...")
    from datasets import load_dataset

    ds = load_dataset("derek-thomas/ScienceQA", split="train")

    prompts = []
    for i, item in enumerate(ds):
        # Filter: text-only (no image)
        if item.get("image") is not None:
            continue
        question = item.get("question", "").strip()
        if not question:
            continue
        choices = item.get("choices", [])
        if choices:
            formatted = format_choices_from_list(choices)
            question = f"{question}\n\nOptions:\n{formatted}"
        # Add context/hint if available
        hint = item.get("hint", "").strip()
        if hint:
            question = f"Context: {hint}\n\n{question}"
        prompts.append({
            "id": f"science_qa_{i:05d}",
            "source": "science_qa",
            "question": question,
            "metadata": {
                "answer": item.get("answer", ""),
                "subject": item.get("subject", ""),
                "topic": item.get("topic", ""),
                "category": item.get("category", ""),
            },
        })

    prompts = sample_items(prompts, target)
    print(f"    Collected {len(prompts)} prompts")
    return prompts


def load_arc(config_name, source_tag, target):
    """allenai/ai2_arc: ARC-Easy or ARC-Challenge."""
    print(f"  Loading allenai/ai2_arc [{config_name}] (target={target})...")
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", config_name, split="train")

    prompts = []
    for i, item in enumerate(ds):
        question = item.get("question", "").strip()
        if not question:
            continue
        choices = item.get("choices", {})
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if labels and texts:
            formatted = format_choices(labels, texts)
            question = f"{question}\n\nOptions:\n{formatted}"
        prompts.append({
            "id": f"{source_tag}_{i:05d}",
            "source": source_tag,
            "question": question,
            "metadata": {
                "answer": item.get("answerKey", ""),
            },
        })

    prompts = sample_items(prompts, target)
    print(f"    Collected {len(prompts)} prompts")
    return prompts


def load_asdiv(target=920):
    """MU-NLPC/Calc-asdiv_a or alternatives: arithmetic word problems."""
    print(f"  Loading asdiv (target={target})...")
    from datasets import load_dataset

    ds = None
    for name in ["MU-NLPC/Calc-asdiv_a", "EleutherAI/asdiv"]:
        try:
            ds = load_dataset(name, split="train", trust_remote_code=True)
            print(f"    Loaded from {name}")
            break
        except Exception as e:
            print(f"    Failed to load {name}: {e}")

    if ds is None:
        # Try test split
        for name in ["MU-NLPC/Calc-asdiv_a", "EleutherAI/asdiv"]:
            try:
                ds = load_dataset(name, split="test", trust_remote_code=True)
                print(f"    Loaded from {name} (test split)")
                break
            except Exception:
                pass

    if ds is None:
        print("    WARNING: Could not load asdiv dataset, skipping")
        return []

    # Inspect first item to understand schema
    first = ds[0]
    print(f"    Schema keys: {list(first.keys())}")

    prompts = []
    for i, item in enumerate(ds):
        # Try different field combos
        body = item.get("body", item.get("question_stem", "")).strip()
        q = item.get("question", item.get("question_text", "")).strip()

        if body and q:
            question = f"{body} {q}"
        elif body:
            question = body
        elif q:
            question = q
        else:
            # Fallback: try 'text' or 'input'
            question = item.get("text", item.get("input", "")).strip()

        if not question or len(question) < 5:
            continue

        prompts.append({
            "id": f"asdiv_{i:05d}",
            "source": "asdiv",
            "question": question,
            "metadata": {
                "answer": str(item.get("answer", item.get("solution", ""))),
            },
        })

    prompts = sample_items(prompts, target)
    print(f"    Collected {len(prompts)} prompts")
    return prompts


def load_med_qa(target=509):
    """bigbio/med_qa: medical question answering."""
    print(f"  Loading med_qa (target={target})...")
    from datasets import load_dataset

    ds = None
    # Try various dataset IDs and configs
    attempts = [
        ("bigbio/med_qa", {"split": "train", "trust_remote_code": True}),
        ("bigbio/med_qa", {"split": "train", "trust_remote_code": True, "name": "med_qa_en_source"}),
        ("bigbio/med_qa", {"split": "train", "trust_remote_code": True, "name": "med_qa_en_bigbio_qa"}),
        ("GBaker/MedQA-USMLE-4-options", {"split": "train"}),
        ("medmcqa", {"split": "train"}),
    ]

    loaded_name = None
    for name, kwargs in attempts:
        try:
            ds = load_dataset(name, **kwargs)
            loaded_name = name
            print(f"    Loaded from {name} (kwargs: {kwargs})")
            break
        except Exception as e:
            print(f"    Failed {name} ({kwargs}): {type(e).__name__}: {e}")

    if ds is None:
        print("    WARNING: Could not load med_qa dataset, skipping")
        return []

    # Inspect schema
    first = ds[0]
    print(f"    Schema keys: {list(first.keys())}")

    prompts = []
    for i, item in enumerate(ds):
        # Handle various schemas
        question = ""

        # bigbio format
        if "question" in item:
            question = item["question"].strip() if isinstance(item["question"], str) else ""
        elif "sent1" in item:
            question = item["sent1"].strip()

        if not question:
            continue

        # Try to get options
        options = item.get("options", item.get("choices", []))
        answer = item.get("answer", item.get("answer_idx", item.get("cop", "")))
        if isinstance(options, dict):
            # bigbio QA format
            labels = options.get("label", options.get("key", []))
            texts = options.get("text", options.get("value", []))
            if labels and texts:
                formatted = format_choices(labels, texts)
                question = f"{question}\n\nOptions:\n{formatted}"
        elif isinstance(options, list) and options:
            if isinstance(options[0], str):
                formatted = format_choices_from_list(options)
                question = f"{question}\n\nOptions:\n{formatted}"
            elif isinstance(options[0], dict):
                labels = [o.get("key", chr(ord("A") + j)) for j, o in enumerate(options)]
                texts = [o.get("value", o.get("text", "")) for o in options]
                formatted = format_choices(labels, texts)
                question = f"{question}\n\nOptions:\n{formatted}"

        # GBaker format: options as separate fields
        if "GBaker" in (loaded_name or ""):
            opts = []
            for key in ["options"]:
                val = item.get(key, {})
                if isinstance(val, dict):
                    for k in sorted(val.keys()):
                        opts.append(f"{k}) {val[k]}")
            if not opts:
                # Try opa, opb, opc, opd (medmcqa style)
                for label, key in zip(["A", "B", "C", "D"], ["opa", "opb", "opc", "opd"]):
                    if key in item:
                        opts.append(f"{label}) {item[key]}")
            if opts and "Options:" not in question:
                question = f"{question}\n\nOptions:\n" + "\n".join(opts)

        # medmcqa format
        if loaded_name == "medmcqa":
            opts = []
            for label, key in zip(["A", "B", "C", "D"], ["opa", "opb", "opc", "opd"]):
                if key in item and item[key]:
                    opts.append(f"{label}) {item[key]}")
            if opts and "Options:" not in question:
                question = f"{question}\n\nOptions:\n" + "\n".join(opts)

        prompts.append({
            "id": f"med_qa_{i:05d}",
            "source": "med_qa",
            "question": question,
            "metadata": {
                "answer": str(answer),
            },
        })

    prompts = sample_items(prompts, target)
    print(f"    Collected {len(prompts)} prompts")
    return prompts


def load_scruples(target=1000):
    """metaeval/scruples: moral dilemmas."""
    print(f"  Loading scruples (target={target})...")
    from datasets import load_dataset

    ds = None
    attempts = [
        ("metaeval/scruples", {"split": "train", "trust_remote_code": True}),
        ("metaeval/scruples", {"split": "train", "trust_remote_code": True, "name": "dilemmas"}),
        ("metaeval/scruples", {"split": "train", "trust_remote_code": True, "name": "anecdotes"}),
    ]

    for name, kwargs in attempts:
        try:
            ds = load_dataset(name, **kwargs)
            print(f"    Loaded from {name} ({kwargs})")
            break
        except Exception as e:
            print(f"    Failed {name} ({kwargs}): {type(e).__name__}: {e}")

    if ds is None:
        print("    WARNING: Could not load scruples dataset, skipping")
        return []

    first = ds[0]
    print(f"    Schema keys: {list(first.keys())}")

    prompts = []
    for i, item in enumerate(ds):
        # Try to extract question/scenario
        question = ""
        for field in ["text", "title", "post", "scenario", "question", "description"]:
            val = item.get(field, "")
            if isinstance(val, str) and val.strip():
                question = val.strip()
                break

        if not question or len(question) < 10:
            continue

        # Extract action description (action can be a dict with 'description' key or a string)
        action_raw = item.get("action", "")
        if isinstance(action_raw, dict):
            action_desc = action_raw.get("description", "").strip()
        elif isinstance(action_raw, str):
            action_desc = action_raw.strip()
        else:
            action_desc = ""

        if action_desc and action_desc not in question:
            question = f"{question}\n\nAction in question: {action_desc}"

        # Add label info to metadata
        label = item.get("label", item.get("gold_label", item.get("binarized_label", "")))

        prompts.append({
            "id": f"scruples_{i:05d}",
            "source": "scruples",
            "question": question,
            "metadata": {
                "label": str(label),
            },
        })

    prompts = sample_items(prompts, target)
    print(f"    Collected {len(prompts)} prompts")
    return prompts


# ─── Local corpus loaders ───────────────────────────────────────────────────────

def load_cot_corpus_v5(target=5000):
    """Load questions from local cot_corpus_v5/corpus_medium.jsonl."""
    path = PROJECT_ROOT / "data" / "cot_corpus_v5" / "corpus_medium.jsonl"
    print(f"  Loading cot_corpus_v5 from {path} (target={target})...")

    if not path.exists():
        print(f"    WARNING: {path} does not exist, skipping")
        return []

    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = item.get("question", "").strip()
            if not question:
                continue
            items.append({
                "id": item.get("id", f"cot_corpus_v5_{len(items):05d}"),
                "source": "cot_corpus_v5",
                "question": question,
                "metadata": {
                    "original_source": item.get("source", ""),
                    "domain": item.get("domain", ""),
                    "subject": item.get("subject", ""),
                    "correct_answer": item.get("correct_answer", ""),
                },
            })

    # Deduplicate by question text
    seen = set()
    deduped = []
    for item in items:
        key = item["question"][:300]
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    items = deduped

    items = sample_items(items, target)
    print(f"    Collected {len(items)} prompts (deduped from corpus)")
    return items


def load_concept_corpus(target=8200):
    """Load questions from local concept_corpus/corpus_full.jsonl."""
    path = PROJECT_ROOT / "data" / "concept_corpus" / "corpus_full.jsonl"
    print(f"  Loading concept_corpus from {path} (target={target})...")

    if not path.exists():
        print(f"    WARNING: {path} does not exist, skipping")
        return []

    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = item.get("question", "").strip()
            if not question:
                continue
            items.append({
                "id": item.get("id", f"concept_corpus_{len(items):05d}"),
                "source": "concept_corpus",
                "question": question,
                "metadata": {
                    "original_source": item.get("source", ""),
                    "domain": item.get("domain", ""),
                    "subject": item.get("subject", ""),
                    "correct_answer": item.get("correct_answer", ""),
                },
            })

    # Deduplicate by question text
    seen = set()
    deduped = []
    for item in items:
        key = item["question"][:300]
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    items = deduped

    items = sample_items(items, target)
    print(f"    Collected {len(items)} prompts (deduped from corpus)")
    return items


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Building diverse rollout prompts")
    print("=" * 70)

    all_prompts = []
    stats = {}

    # HuggingFace datasets
    loaders = [
        ("lmsys", lambda: load_lmsys(20000)),
        ("aqua_rat", lambda: load_aqua_rat(10000)),
        ("mmlu_pro", lambda: load_mmlu_pro(4800)),
        ("commonsense_qa", lambda: load_commonsense_qa(4000)),
        ("gsm8k", lambda: load_gsm8k(3500)),
        ("math", lambda: load_math(3200)),
        ("science_qa", lambda: load_science_qa(1700)),
        ("arc_easy", lambda: load_arc("ARC-Easy", "arc_easy", 950)),
        ("asdiv", lambda: load_asdiv(920)),
        ("med_qa", lambda: load_med_qa(509)),
        ("arc_challenge", lambda: load_arc("ARC-Challenge", "arc_challenge", 469)),
        ("scruples", lambda: load_scruples(1000)),
    ]

    for name, loader in loaders:
        print(f"\n[{name}]")
        try:
            prompts = loader()
            all_prompts.extend(prompts)
            stats[name] = len(prompts)
        except Exception as e:
            print(f"  ERROR loading {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            stats[name] = 0

    # Local corpora
    print(f"\n[cot_corpus_v5]")
    try:
        v5 = load_cot_corpus_v5(5000)
        all_prompts.extend(v5)
        stats["cot_corpus_v5"] = len(v5)
    except Exception as e:
        print(f"  ERROR: {e}")
        stats["cot_corpus_v5"] = 0

    print(f"\n[concept_corpus]")
    try:
        cc = load_concept_corpus(8200)
        all_prompts.extend(cc)
        stats["concept_corpus"] = len(cc)
    except Exception as e:
        print(f"  ERROR: {e}")
        stats["concept_corpus"] = 0

    # Write output
    print(f"\nWriting {len(all_prompts)} prompts to {OUTPUT_FILE}...")
    write_prompts(all_prompts, OUTPUT_FILE)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Source':<25} {'Count':>8}")
    print("-" * 35)
    total = 0
    for name, count in stats.items():
        print(f"{name:<25} {count:>8}")
        total += count
    print("-" * 35)
    print(f"{'TOTAL':<25} {total:>8}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()

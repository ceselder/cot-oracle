"""
CoT Hinted Answer Prediction — Predict the model's final answer from partial hinted CoT.

Given CoT-only activations from a truncated hinted rollout, predict what answer
the model will ultimately produce. Truncation at a random point (25-100% of CoT)
forces the oracle to learn how hints progressively steer reasoning.

This complements the clean answer_pred task: the oracle learns to track answer
formation in both clean and hinted conditions, detecting when a hint is pulling
the model's reasoning toward or away from the hinted answer.

Data source: mats-10-sprint-cs-jb/qwen3-8b-hint-admission-rollouts on HuggingFace.
Only CoT activations are fed (no prompt positions).
"""

import random

from transformers import AutoTokenizer


HF_REPO = "mats-10-sprint-cs-jb/qwen3-8b-hint-admission-rollouts"


def load_cot_hinted_answer_pred_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 10000,
    stride: int | str = None,
    n_prompt_positions: int = 5,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """
    Generate hinted answer prediction training data.

    Each example:
      1. Tokenize hinted prompt + CoT
      2. Truncate CoT at a random point (25-100%)
      3. Get stride positions over the truncated CoT only
      4. Target = the model's final answer (model_answer field)

    Only CoT activations are used (no prompt positions).
    """
    from cot_utils import get_cot_positions, get_injection_layers

    random.seed(seed)

    LAYERS = get_injection_layers(model_name)

    data_path = _kwargs.get("hint_admission_data_path", corpus_path)
    items = _load_data(data_path)

    if not items:
        raise ValueError(
            f"No hint admission data found at {data_path}. "
            f"Download from {HF_REPO}."
        )

    # Question-level split: first 90% for training (same as hint_admission task)
    by_question: dict[str, list[dict]] = {}
    for item in items:
        qid = item.get("question_id", "")
        if qid not in by_question:
            by_question[qid] = []
        by_question[qid].append(item)

    question_ids = sorted(by_question.keys())
    split_rng = random.Random(42)
    split_rng.shuffle(question_ids)
    train_end = int(len(question_ids) * 0.9)
    train_qids = set(question_ids[:train_end])
    items = [item for item in items if item.get("question_id", "") in train_qids]
    print(f"  Question-level split: {len(train_qids)}/{len(question_ids)} questions for training ({len(items)} items)")

    # Filter to items with a valid model_answer
    items = [it for it in items if it.get("model_answer", "").strip()]
    print(f"  Items with valid model_answer: {len(items)}")

    datapoints = []
    attempts = 0

    while len(datapoints) < num_examples and attempts < num_examples * 3:
        attempts += 1
        entry = random.choice(items)

        hinted_prompt = entry.get("hinted_prompt", "")
        if not hinted_prompt:
            continue

        messages = [{"role": "user", "content": hinted_prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        cot_text = entry["cot_text"]
        full_text = formatted + cot_text
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        cot_len = len(full_ids) - prompt_len

        if cot_len < 10:
            continue

        # Random truncation: keep 25-100% of CoT tokens
        trunc_frac = random.uniform(0.25, 1.0)
        trunc_cot_len = max(5, int(cot_len * trunc_frac))
        trunc_total = prompt_len + trunc_cot_len
        trunc_ids = full_ids[:trunc_total]

        positions = get_cot_positions(
            prompt_len, len(trunc_ids),
            stride=stride, tokenizer=tokenizer, input_ids=trunc_ids,
        )
        if len(positions) < 2:
            continue

        # CoT-only positions
        context_positions = positions * len(LAYERS)
        num_positions = len(context_positions)

        max_pos = max(positions)
        context_slice = trunc_ids[:max_pos + 1]

        layers_str = ", ".join(str(l) for l in LAYERS)
        pct = int(trunc_frac * 100)
        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str} "
            f"({pct}% of reasoning). What is the model's final answer?"
        )

        target = entry["model_answer"].strip()

        datapoints.append({
            "datapoint_type": "cot_hinted_answer_pred",
            "prompt": prompt,
            "target_response": target,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} hinted answer prediction examples")
    return datapoints[:num_examples]


def _load_data(path: str) -> list[dict]:
    """Load hint admission data from HuggingFace."""
    from pathlib import Path

    VALID_LABELS = {"hint_used_wrong", "hint_used_correct", "hint_resisted"}

    print(f"  Downloading hinted rollout data from {HF_REPO}...")
    try:
        import importlib
        hf_datasets = importlib.import_module("datasets")
        ds = hf_datasets.load_dataset(HF_REPO, split="train")
        items = []
        for row in ds:
            if row.get("cot_text") and row.get("label") in VALID_LABELS:
                items.append(dict(row))
        if items:
            print(f"  Downloaded {len(items)} items from HuggingFace")
            return items
    except Exception as e:
        print(f"  Warning: HF download failed: {e}")

    # Fallback: local JSONL
    local = Path(path)
    if local.exists() and local.suffix == ".jsonl":
        import json
        items = []
        with open(local) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("cot_text") and item.get("label") in VALID_LABELS:
                        items.append(item)
        if items:
            print(f"  Loaded {len(items)} items from local fallback {local}")
            return items

    return []

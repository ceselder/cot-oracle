"""
CotQA — Conversational questions about CoT activations (concept/safety corpus).

Loads QA pairs from mats-10-sprint-cs-jb/cot-oracle-cotqa and joins with the
concept corpus (mats-10-sprint-cs-jb/cot-oracle-concept-corpus) to get CoT text
for activation extraction.

10 question types: topic, content, general, behavior, style, reasoning,
thematic, sycophancy, safety, user_intent.

This is the canonical conversational QA task (previously also called conv_qa).
"""

import json
import random
import re

from transformers import AutoTokenizer


HF_QA_REPO = "mats-10-sprint-cs-jb/cot-oracle-cotqa"
HF_CORPUS_REPO = "mats-10-sprint-cs-jb/cot-oracle-concept-corpus"


def load_cot_cotqa_data(
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
    Load CotQA training data: conversational questions about safety/concept CoTs.

    Downloads QA pairs + concept corpus from HuggingFace, joins by corpus_id,
    and generates training data with multi-layer stride activations.
    """
    from cot_utils import get_cot_positions, get_injection_layers

    random.seed(seed)

    LAYERS = get_injection_layers(model_name)

    # Load QA pairs and concept corpus from HF
    qa_pairs, corpus_by_id = _load_from_hf()

    if not qa_pairs:
        raise ValueError(f"No CotQA data loaded from {HF_QA_REPO}")
    if not corpus_by_id:
        raise ValueError(f"No concept corpus loaded from {HF_CORPUS_REPO}")

    # Question-level split: first 90% for training, last 10% for eval.
    corpus_ids = sorted(set(qa["corpus_id"] for qa in qa_pairs))
    split_rng = random.Random(42)
    split_rng.shuffle(corpus_ids)
    train_end = int(len(corpus_ids) * 0.9)
    train_ids = set(corpus_ids[:train_end])
    qa_pairs = [qa for qa in qa_pairs if qa["corpus_id"] in train_ids]
    print(f"  Question-level split: {len(train_ids)}/{len(corpus_ids)} corpus entries for training ({len(qa_pairs)} QA pairs)")

    # Filter to pairs with matching corpus entries
    matched = [(qa, corpus_by_id[qa["corpus_id"]]) for qa in qa_pairs if qa["corpus_id"] in corpus_by_id]
    random.shuffle(matched)
    print(f"  Matched {len(matched)} QA pairs to corpus entries")

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    datapoints = []

    for qa, entry in matched:
        if len(datapoints) >= num_examples:
            break

        cot_text = entry.get("cot_response", "")
        if not cot_text.strip():
            continue
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]

        messages = [{"role": "user", "content": entry["question"]}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        full_text = formatted + cot_text
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        positions = get_cot_positions(
            prompt_len, len(full_ids),
            stride=stride, tokenizer=tokenizer, input_ids=full_ids,
        )
        if len(positions) < 2:
            continue

        prompt_positions = _get_prompt_positions(prompt_len, n_prompt_positions)
        combined = prompt_positions + positions
        context_positions = combined * len(LAYERS)
        num_positions = len(context_positions)

        max_pos = max(positions)
        context_slice = full_ids[:max_pos + 1]

        # Extract question text from the QA prompt
        original_prompt = qa.get("prompt", "")
        q_match = re.search(r"(?:sentence boundaries|positions)\.\s*(.*)", original_prompt)
        question_text = q_match.group(1) if q_match else original_prompt

        target = qa["target_response"]
        if not target or not target.strip():
            continue

        layers_str = ", ".join(str(l) for l in LAYERS)
        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}. "
            f"{question_text}"
        )

        datapoints.append({
            "datapoint_type": "cot_cotqa",
            "prompt": prompt,
            "target_response": target,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} CotQA examples "
          f"(from {len(matched)} available pairs)")
    return datapoints[:num_examples]


def _load_from_hf() -> tuple[list[dict], dict[str, dict]]:
    """Load QA pairs and concept corpus from HuggingFace."""
    import importlib
    hf_datasets = importlib.import_module("datasets")

    # Load QA pairs
    print(f"  Downloading CotQA pairs from {HF_QA_REPO}...")
    try:
        qa_ds = hf_datasets.load_dataset(HF_QA_REPO, split="train")
        qa_pairs = [dict(row) for row in qa_ds if row.get("target_response")]
        print(f"  Downloaded {len(qa_pairs)} QA pairs")
    except Exception as e:
        print(f"  Warning: QA download failed: {e}")
        qa_pairs = []

    # Load concept corpus
    print(f"  Downloading concept corpus from {HF_CORPUS_REPO}...")
    corpus_by_id = {}
    try:
        corpus_ds = hf_datasets.load_dataset(HF_CORPUS_REPO, split="train")
        for row in corpus_ds:
            if row.get("cot_response") and row.get("id"):
                corpus_by_id[row["id"]] = dict(row)
        print(f"  Downloaded {len(corpus_by_id)} corpus entries")
    except Exception as e:
        print(f"  Warning: Corpus download failed: {e}")

    return qa_pairs, corpus_by_id

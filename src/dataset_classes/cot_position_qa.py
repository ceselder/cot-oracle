"""
CoT Position QA — Questions about reasoning at specific token positions.

Each example picks a token position in the CoT and asks about the reasoning
function being performed there (using the Thought Anchors taxonomy). The oracle
receives activations only up to that position and must describe what happens next.

Data comes from HuggingFace: position QA pairs + main corpus for original prompts.
"""

import json
import random

from transformers import AutoTokenizer


HF_QA_REPO = "ceselder/cot-oracle-position-qa"
HF_CORPUS_REPO = "mats-10-sprint-cs-jb/cot-oracle-corpus-v5"


def load_cot_position_qa_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 15000,
    stride: int | str = None,
    n_prompt_positions: int = 5,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """
    Load position QA training data.

    Each example:
      1. Joins position QA pair with main corpus entry (by cot_id)
      2. Tokenizes prompt + CoT up to token_position
      3. Extracts stride positions up to token_position
      4. Oracle prompt asks about reasoning at that position
      5. Target is the LLM-generated description of next reasoning steps
    """
    from cot_utils import get_injection_layers

    random.seed(seed)

    LAYERS = get_injection_layers(model_name)

    # Load QA pairs and corpus from HF
    qa_pairs, corpus_by_id = _load_from_hf()

    if not qa_pairs:
        raise ValueError(f"No position QA data loaded from {HF_QA_REPO}")
    if not corpus_by_id:
        raise ValueError(f"No corpus data loaded from {HF_CORPUS_REPO}")

    # 90/10 CoT-level train/eval split (deterministic)
    cot_ids = sorted(set(qa["cot_id"] for qa in qa_pairs))
    split_rng = random.Random(42)
    split_rng.shuffle(cot_ids)
    train_end = int(len(cot_ids) * 0.9)
    train_ids = set(cot_ids[:train_end])
    qa_pairs = [qa for qa in qa_pairs if qa["cot_id"] in train_ids]
    print(f"  CoT-level split: {len(train_ids)}/{len(cot_ids)} CoTs for training "
          f"({len(qa_pairs)} QA pairs)")

    # Filter to pairs with matching corpus entries
    matched = [(qa, corpus_by_id[qa["cot_id"]])
               for qa in qa_pairs if qa["cot_id"] in corpus_by_id]
    random.shuffle(matched)
    print(f"  Matched {len(matched)} QA pairs to corpus entries")

    datapoints = []

    for qa, entry in matched:
        if len(datapoints) >= num_examples:
            break

        answer = qa.get("answer", "")
        if not answer or not answer.strip():
            continue

        cot_text = entry.get("cot_response", "")
        if not cot_text.strip():
            continue
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]

        # Tokenize prompt + CoT
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

        # token_position is relative to CoT start → absolute position
        token_pos = qa["token_position"]
        abs_pos = prompt_len + token_pos

        if abs_pos >= len(full_ids) - 5:
            continue  # position too close to end

        # Only feed the single activation at the target position
        positions = [abs_pos]
        context_positions = positions * len(LAYERS)
        num_positions = len(context_positions)

        # Context: tokens up to the target position
        context_slice = full_ids[:abs_pos + 1]

        # Format oracle prompt
        layers_str = ", ".join(str(l) for l in LAYERS)
        question_text = qa["question"]
        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}. "
            f"{question_text}"
        )

        datapoints.append({
            "datapoint_type": "cot_position_qa",
            "prompt": prompt,
            "target_response": answer,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} position QA examples "
          f"(from {len(matched)} available pairs)")
    return datapoints[:num_examples]


def _load_from_hf() -> tuple[list[dict], dict[str, dict]]:
    """Load position QA pairs and main corpus from HuggingFace."""
    import importlib
    hf_datasets = importlib.import_module("datasets")

    # Load position QA pairs
    print(f"  Downloading position QA pairs from {HF_QA_REPO}...")
    try:
        qa_ds = hf_datasets.load_dataset(HF_QA_REPO, split="train")
        qa_pairs = [dict(row) for row in qa_ds if row.get("answer")]
        print(f"  Downloaded {len(qa_pairs)} QA pairs")
    except Exception as e:
        print(f"  Warning: QA download failed: {e}")
        qa_pairs = []

    # Load main corpus
    print(f"  Downloading corpus from {HF_CORPUS_REPO}...")
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

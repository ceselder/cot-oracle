"""Activation extraction & data loading for baselines."""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.ao import (
    EarlyStopException,
    generate_cot,
    get_hf_submodule,
    using_adapter,
)
from cot_utils import get_cot_stride_positions, split_cot_into_sentences, find_sentence_boundary_positions
from evals.common import (
    EvalItem,
    load_eval_items_hf,
    determine_ground_truth,
    extract_letter_answer,
    extract_numerical_answer,
)
from evals.activation_cache import build_full_text_from_prompt_and_cot


@dataclass
class BaselineInput:
    eval_name: str
    example_id: str
    clean_prompt: str
    test_prompt: str
    correct_answer: str
    nudge_answer: str | None
    ground_truth_label: str
    clean_response: str
    test_response: str
    activations_by_layer: dict[int, torch.Tensor]  # {layer: [K, D]}
    metadata: dict = field(default_factory=dict)


def _extract_answer(response: str, eval_name: str) -> str | None:
    """Extract answer from model response (mirrors run_evals.py logic)."""
    if eval_name in ("hinted_mcq", "hinted_mcq_truthfulqa"):
        return extract_letter_answer(response)
    elif eval_name == "sycophancy_v2_riya":
        lower = response.lower() if response else ""
        matches = list(re.finditer(r"\b(right|wrong)\b", lower))
        return matches[-1].group(1).upper() if matches else None
    return extract_numerical_answer(response)


def extract_multilayer_activations(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    positions: list[int], layers: list[int],
) -> dict[int, torch.Tensor]:
    """Single forward pass extracting activations at positions from multiple layers.

    Returns {layer: [K, D]} where K = len(positions), on CPU.
    """
    submodules = {l: get_hf_submodule(model, l) for l in layers}
    max_layer = max(layers)
    acts: dict[int, torch.Tensor] = {}
    mod_to_layer = {id(s): l for l, s in submodules.items()}

    def hook_fn(module, _inputs, outputs):
        layer = mod_to_layer[id(module)]
        raw = outputs[0] if isinstance(outputs, tuple) else outputs
        acts[layer] = raw[0, positions, :].detach().cpu()
        if layer == max_layer:
            raise EarlyStopException()

    handles = [s.register_forward_hook(hook_fn) for s in submodules.values()]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    except EarlyStopException:
        pass
    finally:
        for h in handles:
            h.remove()
    return acts


def load_baseline_inputs(
    eval_name: str,
    model,
    tokenizer,
    *,
    layers: list[int],
    stride: int,
    device: str = "cuda",
    eval_dir: Path | None = None,
    generation_adapter_name: str | None = None,
) -> list[BaselineInput]:
    """Load eval items, run model for responses, extract multi-layer activations.

    Filters out items with indeterminate ground truth.
    """
    items = load_eval_items_hf(eval_name, eval_dir=eval_dir)
    print(f"  Loaded {len(items)} items for {eval_name}")

    is_cls = eval_name.startswith("cls_")

    results = []
    was_training = model.training
    model.eval()

    for item in tqdm(items, desc=f"Loading {eval_name}"):
        if is_cls:
            # Classification evals: neutral prompt, text from metadata, ground truth = correct_answer
            cot_text = item.metadata.get("cot_text", "")
            if not cot_text:
                continue
            clean_response = test_response = cot_text
            gt = item.correct_answer  # "yes" or "no"
            extraction_prompt = "Read the following text."
        elif eval_name == "sentence_insertion":
            # Use spliced CoT text, binary ground truth
            cot_text = item.metadata.get("spliced_cot_text", "")
            if not cot_text:
                continue
            clean_response = test_response = cot_text
            gt = "inserted" if item.metadata.get("is_insertion") else "clean"
            extraction_prompt = item.test_prompt
        else:
            # Standard evals: use precomputed responses if available, otherwise generate
            precomp_clean = item.metadata.get("qwen3_8b_clean_response")
            precomp_test = item.metadata.get("qwen3_8b_test_response")
            if precomp_clean and precomp_test:
                clean_response, test_response = precomp_clean, precomp_test
                clean_answer = item.metadata.get("qwen3_8b_clean_answer") or _extract_answer(clean_response, eval_name)
                test_answer = item.metadata.get("qwen3_8b_test_answer") or _extract_answer(test_response, eval_name)
            else:
                clean_response = generate_cot(model, tokenizer, item.clean_prompt, max_new_tokens=512, device=device, adapter_name=generation_adapter_name)
                test_response = generate_cot(model, tokenizer, item.test_prompt, max_new_tokens=512, device=device, adapter_name=generation_adapter_name)
                clean_answer = _extract_answer(clean_response, eval_name)
                test_answer = _extract_answer(test_response, eval_name)

            # Determine ground truth
            gt = determine_ground_truth(item, clean_answer, test_answer)
            if eval_name == "decorative_cot":
                gt = item.metadata.get("ground_truth_label") or item.metadata.get("label") or gt
            elif eval_name in ("thought_anchors", "thought_branches"):
                gt = "ranking"
            elif eval_name in ("rot13_reconstruction", "chunked_convqa", "cot_hint_admission"):
                gt = "generation"

            cot_text = test_response.strip()
            extraction_prompt = item.test_prompt

        skip_labels = {"indeterminate", "pending_multi_run", "pending_manual",
                       "pending_reconstruction", "pending_entropy_regression"}
        if gt in skip_labels:
            continue

        # Build full text for activation extraction
        cot_text = cot_text.strip()
        if not cot_text:
            continue

        full_text = build_full_text_from_prompt_and_cot(tokenizer, extraction_prompt, cot_text)
        messages = [{"role": "user", "content": extraction_prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        all_ids = tokenizer.encode(full_text, add_special_tokens=False)
        positions = get_cot_stride_positions(len(prompt_ids), len(all_ids), stride=stride)

        if len(positions) < 2:
            continue

        tok_out = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        input_tensor = tok_out["input_ids"].to(device)
        attn_mask = tok_out["attention_mask"].to(device)
        seq_len = input_tensor.shape[1]
        positions = [p for p in positions if p < seq_len]
        if len(positions) < 2:
            continue

        # Extract activations from all layers in one pass (base model, adapters disabled)
        with using_adapter(model, None):
            acts_by_layer = extract_multilayer_activations(model, input_tensor, attn_mask, positions, layers)

        meta = {**item.metadata, "positions": positions}

        # For ranking evals, compute chunk-to-position mapping
        if eval_name in ("thought_anchors", "thought_branches"):
            chunks = item.metadata.get("cot_chunks", split_cot_into_sentences(cot_text))
            boundaries = find_sentence_boundary_positions(tokenizer, full_text, chunks)
            chunk_pos_indices = []
            for bp in boundaries:
                nearest_idx = min(range(len(positions)), key=lambda i: abs(positions[i] - bp))
                chunk_pos_indices.append(nearest_idx)
            meta["chunk_position_indices"] = chunk_pos_indices
            meta["cot_chunks"] = chunks

        results.append(BaselineInput(
            eval_name=eval_name, example_id=item.example_id,
            clean_prompt=item.clean_prompt, test_prompt=item.test_prompt,
            correct_answer=item.correct_answer, nudge_answer=item.nudge_answer,
            ground_truth_label=gt, clean_response=clean_response, test_response=test_response,
            activations_by_layer=acts_by_layer, metadata=meta,
        ))

    if was_training:
        model.train()
    print(f"  {len(results)} usable items (filtered from {len(items)})")
    return results


CLEANED_DATASET_NAMES = {
    "mats-10-sprint-cs-jb/cot-oracle-correctness-cleaned": "cleaned_correctness",
    "mats-10-sprint-cs-jb/cot-oracle-reasoning-termination-cleaned": "cleaned_reasoning_termination",
    "mats-10-sprint-cs-jb/cot-oracle-atypical-answer-cleaned": "cleaned_atypical_answer",
    "mats-10-sprint-cs-jb/cot-oracle-hint-admission-cleaned": "cleaned_hint_admission",
    "mats-10-sprint-cs-jb/cot-oracle-truthfulqa-hint-unverbalized-cleaned": "cleaned_truthfulqa_hint_unverb",
    "mats-10-sprint-cs-jb/cot-oracle-truthfulqa-hint-verbalized-cleaned": "cleaned_truthfulqa_hint_verb",
}


def load_cleaned_baseline_inputs(
    dataset_id: str, model, tokenizer, *,
    layers: list[int], stride: int, device: str = "cuda",
    train_fraction: float = 0.01,
) -> tuple[list[BaselineInput], list[BaselineInput]]:
    """Load cleaned HF dataset with train/test splits, extract activations.

    Returns (train_inputs, test_inputs).
    """
    eval_name = CLEANED_DATASET_NAMES[dataset_id]
    ds = load_dataset(dataset_id)
    train_split = ds["train"]
    test_split = ds["test"]

    if train_fraction < 1.0:
        n_total = len(train_split)
        n_used = max(1, int(n_total * train_fraction))
        print(f"WARNING: Using {train_fraction*100:.0f}% of training data ({n_used}/{n_total} rows). Set train_fraction=1.0 for full training.")
        train_split = train_split.shuffle(seed=42).select(range(n_used))

    was_training = model.training
    model.eval()

    def _process_split(split, split_name):
        results = []
        for i, row in enumerate(tqdm(split, desc=f"Loading {eval_name} {split_name}")):
            cot_text = row["cot_text"].strip()
            if not cot_text:
                continue
            prompt = row.get("hinted_prompt") or row.get("question", "")

            full_text = build_full_text_from_prompt_and_cot(tokenizer, prompt, cot_text)
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
            all_ids = tokenizer.encode(full_text, add_special_tokens=False)
            positions = get_cot_stride_positions(len(prompt_ids), len(all_ids), stride=stride)

            if len(positions) < 2:
                continue

            tok_out = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
            input_tensor = tok_out["input_ids"].to(device)
            attn_mask = tok_out["attention_mask"].to(device)
            seq_len = input_tensor.shape[1]
            positions = [p for p in positions if p < seq_len]
            if len(positions) < 2:
                continue

            with using_adapter(model, None):
                acts_by_layer = extract_multilayer_activations(model, input_tensor, attn_mask, positions, layers)

            results.append(BaselineInput(
                eval_name=eval_name,
                example_id=row.get("question_id", f"{dataset_id}_{i}"),
                clean_prompt=prompt, test_prompt=prompt,
                correct_answer=row.get("correct_answer", ""),
                nudge_answer=None,
                ground_truth_label=row["label"],
                clean_response=cot_text, test_response=cot_text,
                activations_by_layer=acts_by_layer,
                metadata={"positions": positions},
            ))
        return results

    train_inputs = _process_split(train_split, "train")
    test_inputs = _process_split(test_split, "test")

    if was_training:
        model.train()
    print(f"  {len(train_inputs)} train, {len(test_inputs)} test items for {eval_name}")
    return train_inputs, test_inputs


def pool_activations(acts: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Pool [K, D] -> [D]."""
    if method == "mean": return acts.mean(dim=0)
    elif method == "max": return acts.max(dim=0).values
    elif method == "last": return acts[-1]
    raise ValueError(f"Unknown pooling: {method}")


def log_results(
    results: dict, eval_name: str, baseline_name: str,
    output_dir: Path, log_dir: Path, wandb_run=None,
) -> None:
    """Save results to JSON on disk and log to wandb."""
    out_path = Path(log_dir) / eval_name / f"{baseline_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved {baseline_name} results -> {out_path}")

    if wandb_run is None:
        return

    import wandb
    prefix = f"baselines/{eval_name}/{baseline_name}"
    flat = {}
    for k, v in results.items():
        if isinstance(v, (int, float)):
            flat[f"{prefix}/{k}"] = v
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, (int, float)):
                    flat[f"{prefix}/{k}/{k2}"] = v2
    wandb_run.log(flat)

    # Log sample traces as wandb table
    if "traces" in results and results["traces"]:
        cols = list(results["traces"][0].keys())
        table = wandb.Table(columns=cols)
        for trace in results["traces"][:50]:
            table.add_data(*[trace.get(c, "") for c in cols])
        wandb_run.log({f"{prefix}/traces": table})

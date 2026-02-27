#!/usr/bin/env python
"""Compare Activation Oracle vs SAE feature interpretations on the same CoT rollouts.

Both the trained oracle and the SAEs read from the same residual stream layers [9, 18, 27].
This script generates CoT with the base model, then:
  1. Extracts activations at stride positions
  2. Runs SAE encoding to find top-K active features per position per layer
  3. Queries the trained oracle on multiple tasks (recon, domain, correctness, answer, decorative)
  4. Writes per-example markdown logs and a summary JSON

Usage:
    python scripts/compare_oracle_sae.py \
        --checkpoint checkpoints/final \
        --sae-labels-dir /ceph/scratch/jbauer/sae_features/trainer_2/trainer_2/labels \
        --n-corpus 15

    python scripts/compare_oracle_sae.py \
        --checkpoint checkpoints/final \
        --sae-labels-dir /ceph/scratch/jbauer/sae_features/trainer_2/trainer_2/labels \
        --questions-file questions.txt
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm.auto import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "ao_reference"))

from chat_compare import (
    TASK_PROMPTS,
    collect_multilayer_activations,
    compute_layers,
    generate_cot_base,
    load_dual_model,
    query_trained_oracle,
)
from cot_utils import get_cot_positions
from nl_probes.sae import load_dictionary_learning_batch_topk_sae

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path.home() / ".env")

MODEL_NAME = "Qwen/Qwen3-8B"
SAE_REPO = "adamkarvonen/qwen3-8b-saes"

DEFAULT_QUESTIONS = [
    "What is 127 * 43?",
    "A train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours. What is the total distance?",
    "Why does ice float on water?",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "What would happen if the Earth suddenly stopped rotating?",
    "Is it ethical to use animals in medical research that could save human lives?",
    "Explain the difference between correlation and causation with an example.",
    "A farmer has 17 sheep. All but 9 die. How many are left?",
    "What are the main causes of the French Revolution?",
]


# --- Loading ---

def load_saes(sae_repo, trainer, layers, device):
    """Load SAEs for each layer."""
    saes = {}
    sae_local_dir = str(PROJECT_ROOT / "downloaded_saes")
    for layer in layers:
        filename = f"saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_{layer}/trainer_{trainer}/ae.pt"
        print(f"Loading SAE layer {layer}...")
        saes[layer] = load_dictionary_learning_batch_topk_sae(
            repo_id=sae_repo, filename=filename, model_name=MODEL_NAME,
            device=torch.device(device), dtype=torch.bfloat16, layer=layer,
            local_dir=sae_local_dir,
        )
    return saes


def load_sae_labels(labels_dir, trainer, layers):
    """Load SAE label JSON files for each layer."""
    labels = {}
    for layer in layers:
        path = Path(labels_dir) / f"labels_layer{layer}_trainer{trainer}.json"
        print(f"Loading labels: {path}")
        data = json.loads(path.read_text())
        labels[layer] = data
    return labels


def load_all(model_name, checkpoint, sae_repo, trainer, labels_dir, layers, device):
    """Load base model + LoRA adapters + SAEs + label dicts."""
    model, tokenizer = load_dual_model(model_name, checkpoint, device=device)
    saes = load_saes(sae_repo, trainer, layers, device)
    labels = load_sae_labels(labels_dir, trainer, layers)
    return model, tokenizer, saes, labels


# --- Activation extraction (single forward pass via hooks) ---

def extract_activations_all_layers(model, tokenizer, full_text, layers, device):
    """Register hooks on all target layers, single forward pass, return {layer: [1, L, D]}.

    Adapters disabled — matches oracle training conditions.
    """
    activation_cache = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            activation_cache[layer_idx] = h.detach()
        return hook_fn

    # Register hooks
    handles = []
    for layer_idx in layers:
        # Navigate through PeftModel wrapper to get the actual layer
        paths_to_try = [
            lambda l=layer_idx: model.base_model.model.model.layers[l],
            lambda l=layer_idx: model.base_model.model.layers[l],
            lambda l=layer_idx: model.model.model.layers[l],
            lambda l=layer_idx: model.model.layers[l],
        ]
        submodule = None
        for path_fn in paths_to_try:
            try:
                submodule = path_fn()
                break
            except (AttributeError, IndexError):
                continue
        assert submodule is not None, f"Could not find layer {layer_idx}"
        handles.append(submodule.register_forward_hook(make_hook(layer_idx)))

    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad(), model.disable_adapter():
        model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    for h in handles:
        h.remove()

    return activation_cache  # {layer: [1, L, D]}


# --- SAE analysis ---

def get_sae_top_features(saes, labels, layer_acts, positions, tokenizer, input_ids, top_k=10):
    """Encode activations at stride positions with SAEs, return top-K features per position.

    Args:
        layer_acts: {layer: [1, L, D]} full sequence activations
        positions: list of token positions in CoT

    Returns:
        {layer: [{pos, token, token_text, features: [(idx, act_val, label), ...]}]}
    """
    results = {}
    for layer, acts_1LD in layer_acts.items():
        if layer not in saes:
            continue
        sae = saes[layer]
        layer_labels = labels.get(layer, {})

        # Index stride positions: [K, D]
        acts_at_pos = acts_1LD[0, positions, :]  # [K, D]

        # SAE encode: expects [B, L, D] or [B, D]
        sae_acts = sae.encode(acts_at_pos)  # [K, F]

        layer_results = []
        for i, pos in enumerate(positions):
            feat_acts = sae_acts[i]  # [F]
            # Get top-K active features
            topk = feat_acts.topk(top_k)
            features = []
            for val, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                if val <= 0:
                    continue
                label_info = layer_labels.get(str(idx), {})
                label_text = label_info.get("label", "unlabeled")
                features.append((idx, val, label_text))

            token_text = tokenizer.decode([input_ids[pos]])
            layer_results.append({
                "pos": pos,
                "token_id": input_ids[pos],
                "token_text": token_text,
                "features": features,
            })

        results[layer] = layer_results
    return results


# --- Aggregate ---

def aggregate_features(sae_results, top_k=20):
    """Per-layer: feature frequency counts + mean activation across all positions.

    Returns: {layer: [(feat_idx, count, mean_act, label), ...]} sorted by count desc.
    """
    aggregated = {}
    for layer, position_data in sae_results.items():
        feat_counts = Counter()
        feat_sum_act = Counter()
        feat_labels = {}
        for entry in position_data:
            for feat_idx, act_val, label in entry["features"]:
                feat_counts[feat_idx] += 1
                feat_sum_act[feat_idx] += act_val
                feat_labels[feat_idx] = label

        ranked = []
        for feat_idx, count in feat_counts.most_common(top_k):
            mean_act = feat_sum_act[feat_idx] / count
            ranked.append((feat_idx, count, mean_act, feat_labels[feat_idx]))
        aggregated[layer] = ranked
    return aggregated


# --- Log writing ---

def write_example_log(output_dir, idx, question, cot_text, cot_tokens, n_positions,
                      oracle_outputs, sae_results, aggregated, stride):
    """Write per-example markdown log."""
    path = output_dir / f"example_{idx:03d}.md"
    lines = []
    lines.append(f"# Example {idx}: {question}\n")

    lines.append(f"## CoT ({cot_tokens} tokens, {n_positions} stride positions, stride={stride})\n")
    lines.append(cot_text)
    lines.append("")

    lines.append("## Oracle Outputs\n")
    for task, response in oracle_outputs.items():
        lines.append(f"### {task}")
        lines.append(response)
        lines.append("")

    # SAE features at stride positions (show first 10 positions per layer)
    lines.append("## SAE Features at Stride Positions\n")
    for layer in sorted(sae_results.keys()):
        positions = sae_results[layer]
        lines.append(f"### Layer {layer}\n")
        for entry in positions[:10]:  # First 10 positions
            token_text = entry["token_text"].replace("|", "\\|")
            lines.append(f"**Position {entry['pos']}** (token: `{token_text}`)\n")
            if entry["features"]:
                lines.append("| Rank | Feature | Act | Label |")
                lines.append("|------|---------|-----|-------|")
                for rank, (feat_idx, act_val, label) in enumerate(entry["features"], 1):
                    label_clean = label.replace("|", "\\|")
                    lines.append(f"| {rank} | {feat_idx} | {act_val:.1f} | {label_clean} |")
            else:
                lines.append("*(no active features)*")
            lines.append("")
        if len(positions) > 10:
            lines.append(f"*... {len(positions) - 10} more positions omitted*\n")

    # Aggregate features
    lines.append("## Aggregate SAE Features\n")
    for layer in sorted(aggregated.keys()):
        ranked = aggregated[layer]
        n_pos = len(sae_results[layer])
        lines.append(f"### Layer {layer} (top features by frequency)\n")
        lines.append(f"| Feature | Freq/{n_pos} | Mean Act | Label |")
        lines.append("|---------|------------|----------|-------|")
        for feat_idx, count, mean_act, label in ranked:
            label_clean = label.replace("|", "\\|")
            lines.append(f"| {feat_idx} | {count} | {mean_act:.1f} | {label_clean} |")
        lines.append("")

    path.write_text("\n".join(lines))
    return path


# --- Main pipeline ---

def analyze_example(model, tokenizer, saes, labels, question, layers, stride,
                    oracle_tasks, top_k_features, top_k_aggregate,
                    max_cot_tokens, max_oracle_tokens, device):
    """Full pipeline for one example: generate CoT -> extract acts -> SAE features -> oracle queries."""
    # 1. Generate CoT with base model (no adapter)
    print(f"  Generating CoT...")
    cot_response = generate_cot_base(model, tokenizer, question,
                                     max_new_tokens=max_cot_tokens, device=device)

    # 2. Tokenize to compute stride positions
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    full_text = formatted + cot_response
    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    all_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    total_len = len(all_ids)
    cot_tokens = total_len - prompt_len

    stride_positions = get_cot_positions(prompt_len, total_len, stride=stride,
                                         tokenizer=tokenizer, input_ids=all_ids)
    n_positions = len(stride_positions)
    print(f"  CoT: {cot_tokens} tokens, {n_positions} stride positions")

    if n_positions < 2:
        print(f"  WARNING: Too few stride positions ({n_positions}), skipping")
        return None

    # 3. Extract activations from all layers in one forward pass
    print(f"  Extracting activations ({len(layers)} layers)...")
    layer_acts = extract_activations_all_layers(model, tokenizer, full_text, layers, device)

    # 4a. Collect multilayer activations for oracle (via existing function)
    print(f"  Collecting oracle activations...")
    multilayer_acts = collect_multilayer_activations(
        model, tokenizer, full_text, layers, stride_positions, device=device,
    )

    # 4b. SAE analysis
    print(f"  Running SAE encoding...")
    sae_results = get_sae_top_features(
        saes, labels, layer_acts, stride_positions,
        tokenizer, all_ids, top_k=top_k_features,
    )
    aggregated = aggregate_features(sae_results, top_k=top_k_aggregate)

    # 5. Query oracle on each task
    oracle_outputs = {}
    for task in tqdm(oracle_tasks, desc="  Oracle tasks", leave=False):
        prompt = TASK_PROMPTS[task]
        print(f"  Oracle: {task}...")
        response = query_trained_oracle(
            model, tokenizer, multilayer_acts, prompt,
            model_name=MODEL_NAME, layers=layers,
            max_new_tokens=max_oracle_tokens, device=device,
        )
        oracle_outputs[task] = response

    return {
        "question": question,
        "cot_response": cot_response,
        "cot_tokens": cot_tokens,
        "n_positions": n_positions,
        "stride": stride,
        "oracle_outputs": oracle_outputs,
        "sae_results": {
            layer: [
                {"pos": e["pos"], "token_text": e["token_text"],
                 "features": [(idx, val, lbl) for idx, val, lbl in e["features"]]}
                for e in entries
            ]
            for layer, entries in sae_results.items()
        },
        "aggregated": {
            layer: [(idx, cnt, mean, lbl) for idx, cnt, mean, lbl in ranked]
            for layer, ranked in aggregated.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Activation Oracle vs SAE features")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--checkpoint", default="checkpoints/final", help="Trained LoRA checkpoint")
    parser.add_argument("--sae-repo", default=SAE_REPO)
    parser.add_argument("--sae-trainer", type=int, default=2)
    parser.add_argument("--sae-labels-dir", required=True, help="Directory with labels_layer*_trainer*.json")
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--questions-file", default=None, help="File with one question per line")
    parser.add_argument("--n-corpus", type=int, default=None, help="Sample N questions from training corpus")
    parser.add_argument("--oracle-tasks", nargs="+", default=["recon", "domain", "correctness", "answer", "decorative"])
    parser.add_argument("--top-k-features", type=int, default=10)
    parser.add_argument("--top-k-aggregate", type=int, default=20)
    parser.add_argument("--max-cot-tokens", type=int, default=4096)
    parser.add_argument("--max-oracle-tokens", type=int, default=300)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    layers = compute_layers(args.model, layers=args.layers)
    print(f"Layers: {layers}")

    # Determine questions
    if args.questions_file:
        questions = Path(args.questions_file).read_text().strip().splitlines()
        questions = [q.strip() for q in questions if q.strip()]
    elif args.n_corpus:
        from datasets import load_dataset
        ds = load_dataset("mats-10-sprint-cs-jb/cot-oracle-corpus-v5", split="train")
        import random
        indices = random.sample(range(len(ds)), min(args.n_corpus, len(ds)))
        questions = [ds[i]["question"] for i in indices]
    else:
        questions = DEFAULT_QUESTIONS

    print(f"Questions: {len(questions)}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "eval_logs" / "oracle_vs_sae" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Load everything
    model, tokenizer, saes, labels = load_all(
        args.model, args.checkpoint, args.sae_repo, args.sae_trainer,
        args.sae_labels_dir, layers, args.device,
    )

    # Run pipeline
    all_results = []
    for idx, question in enumerate(tqdm(questions, desc="Examples")):
        print(f"\n{'='*60}")
        print(f"Example {idx}: {question[:80]}")
        print(f"{'='*60}")

        result = analyze_example(
            model, tokenizer, saes, labels, question, layers, args.stride,
            args.oracle_tasks, args.top_k_features, args.top_k_aggregate,
            args.max_cot_tokens, args.max_oracle_tokens, args.device,
        )

        if result is None:
            print(f"  Skipped (too few positions)")
            continue

        # Write per-example log
        log_path = write_example_log(
            output_dir, idx, question,
            result["cot_response"], result["cot_tokens"], result["n_positions"],
            result["oracle_outputs"], result["sae_results"], result["aggregated"],
            result["stride"],
        )
        print(f"  Wrote: {log_path}")

        # Store for summary (convert tuples to serializable format)
        summary_entry = {
            "idx": idx,
            "question": result["question"],
            "cot_tokens": result["cot_tokens"],
            "n_positions": result["n_positions"],
            "oracle_outputs": result["oracle_outputs"],
            "top_sae_features": {
                str(layer): [
                    {"feature": idx_, "count": cnt, "mean_act": round(mean, 2), "label": lbl}
                    for idx_, cnt, mean, lbl in ranked[:5]
                ]
                for layer, ranked in result["aggregated"].items()
            },
        }
        all_results.append(summary_entry)

    # Write summary
    summary_path = output_dir / "summary.json"
    summary = {
        "config": {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "layers": layers,
            "stride": args.stride,
            "oracle_tasks": args.oracle_tasks,
            "top_k_features": args.top_k_features,
            "n_questions": len(questions),
        },
        "results": all_results,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary: {summary_path}")
    print(f"Logs: {output_dir}")
    print(f"Done! {len(all_results)} examples processed.")


if __name__ == "__main__":
    main()

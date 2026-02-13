"""
AO accuracy vs prediction distance — TEACHER FORCING version.

Instead of autoregressive generation, we feed the actual ground truth tokens
and check the log probability of each correct token at each distance.
No error compounding. Gives smooth probability curves.

Also runs a baseline (no activation injected) to measure floor.

Usage:
    python3 src/ao_distance_accuracy_tf.py \
        --model Qwen/Qwen3-8B \
        --max-distance 50 \
        --n-samples 200

Outputs:
    results/ao_distance_tf/accuracy_vs_distance.png
    results/ao_distance_tf/accuracy_vs_distance.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download

# ── Constants ──────────────────────────────────────────────

LAYER_COUNTS = {
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-8B": 36,
}

AO_CHECKPOINTS = {
    "Qwen/Qwen3-1.7B": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B",
    "Qwen/Qwen3-8B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
}

SPECIAL_TOKEN = " ?"


def layer_percent_to_layer(model_name, pct):
    return int(LAYER_COUNTS[model_name] * pct / 100)


# ── Model Loading ──────────────────────────────────────────

def load_model(model_name, device="cuda"):
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs = {"device_map": device, "torch_dtype": dtype, "attn_implementation": "sdpa"}
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    ao_repo = AO_CHECKPOINTS[model_name]
    ao_path = snapshot_download(ao_repo)
    print(f"Loading AO from {ao_path}...")
    model = PeftModel.from_pretrained(model, ao_path, is_trainable=False)
    model.eval()
    print(f"  Adapters: {list(model.peft_config.keys())}")
    return model, tokenizer


# ── Activation Collection ──────────────────────────────────

class EarlyStop(Exception):
    pass


def get_layer_module(model, layer):
    for path_fn in [
        lambda: model.base_model.model.model.layers[layer],
        lambda: model.model.model.layers[layer],
        lambda: model.model.layers[layer],
    ]:
        try:
            return path_fn()
        except (AttributeError, IndexError):
            continue
    raise ValueError(f"Cannot find layer {layer}")


def collect_activation_at_position(model, input_ids, attn_mask, layer, position):
    activation = None
    submodule = get_layer_module(model, layer)

    def hook(mod, inp, out):
        nonlocal activation
        activation = (out[0] if isinstance(out, tuple) else out)
        raise EarlyStop()

    with model.disable_adapter():
        handle = submodule.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn_mask)
        except EarlyStop:
            pass
        finally:
            handle.remove()

    return activation[0, position, :].detach()


# ── Teacher Forcing Oracle Query ───────────────────────────

def query_ao_teacher_forcing(model, tokenizer, activations, ground_truth_ids,
                             layer, direction="future", device="cuda"):
    """
    Feed ground truth tokens to AO with teacher forcing.
    Returns log probabilities and top-1 correct flags for each position.

    activations: [num_positions, d_model] tensor, or None (for baseline)
    layer: layer number for the prompt prefix
    ground_truth_ids: list of token IDs to evaluate against
    """
    dtype = torch.bfloat16
    k_tokens = len(ground_truth_ids)

    if activations is not None:
        num_positions = activations.shape[0]
    else:
        num_positions = 1  # still need a placeholder for prompt format

    if direction == "future":
        prompt_text = f"Can you predict the next {k_tokens} tokens that come after this?"
    else:
        prompt_text = f"Can you predict the previous {k_tokens} tokens that came before this?"

    prefix = f"Layer: {layer}\n" + SPECIAL_TOKEN * num_positions + " \n"
    full_prompt = prefix + prompt_text

    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)

    # Find placeholder positions
    special_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
    assert len(special_id) == 1
    special_id = special_id[0]

    placeholder_positions = []
    for i, tid in enumerate(prompt_ids):
        if tid == special_id and len(placeholder_positions) < num_positions:
            placeholder_positions.append(i)
    assert len(placeholder_positions) == num_positions

    # Concatenate prompt + ground truth tokens
    full_ids = prompt_ids + ground_truth_ids
    input_tensor = torch.tensor([full_ids], device=device)
    attn_mask = torch.ones_like(input_tensor)

    # Steering hook (or no-op for baseline)
    inject_module = get_layer_module(model, 1)

    if activations is not None:
        normed = F.normalize(activations, dim=-1).detach()  # [num_positions, D]

        def steering_hook(module, _input, output):
            if isinstance(output, tuple):
                resid, *rest = output
                is_tuple = True
            else:
                resid = output
                is_tuple = False

            B, L, D = resid.shape
            if L <= 1:
                return (resid, *rest) if is_tuple else resid

            for k, pos in enumerate(placeholder_positions):
                orig = resid[0, pos, :]
                norm_val = orig.norm()
                steered = (normed[k].to(device, dtype) * norm_val).detach()
                resid[0, pos, :] = steered + orig
            return (resid, *rest) if is_tuple else resid

        handle = inject_module.register_forward_hook(steering_hook)
    else:
        handle = None

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=attn_mask)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
    finally:
        if handle is not None:
            handle.remove()

    # Extract predictions at each distance
    prompt_len = len(prompt_ids)
    log_probs = []
    top1_correct = []
    top5_correct = []

    for d in range(k_tokens):
        pred_pos = prompt_len - 1 + d
        target_id = ground_truth_ids[d]

        if pred_pos >= logits.shape[0]:
            break

        token_logits = logits[pred_pos]
        probs = F.softmax(token_logits, dim=-1)
        log_prob = torch.log(probs[target_id] + 1e-10).item()

        top1 = token_logits.argmax().item() == target_id
        top5_ids = token_logits.topk(5).indices.tolist()
        top5 = target_id in top5_ids

        log_probs.append(log_prob)
        top1_correct.append(top1)
        top5_correct.append(top5)

    return {
        "log_probs": log_probs,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
    }


# ── Data Loading ───────────────────────────────────────────

def load_fineweb_samples(tokenizer, n_samples, min_length=200, seed=42):
    from datasets import load_dataset
    print("Loading FineWeb samples...")
    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)

    samples = []
    rng = random.Random(seed)
    for item in ds:
        text = item["text"]
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= min_length:
            samples.append(ids)
        if len(samples) >= n_samples * 3:
            break

    rng.shuffle(samples)
    samples = samples[:n_samples]
    print(f"  Got {len(samples)} FineWeb samples (min {min_length} tokens)")
    return samples


def load_cot_samples(corpus_path, tokenizer, n_samples, min_length=200, seed=42):
    print(f"Loading CoT samples from {corpus_path}...")
    entries = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(entries)

    samples = []
    for entry in entries:
        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        full_text = formatted + entry["cot_response"]
        ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(ids) >= min_length:
            samples.append(ids)
        if len(samples) >= n_samples:
            break

    print(f"  Got {len(samples)} CoT samples (min {min_length} tokens)")
    return samples


# ── Main Experiment ────────────────────────────────────────

def run_tf_experiment(model, tokenizer, token_id_samples, max_distance, layers,
                      direction="future", use_activation=True, prompt_layer=None,
                      label_str="AO", device="cuda"):
    """
    Teacher forcing experiment. For each sample:
    1. Pick random position P
    2. Collect activation(s) at P from each layer in `layers` (or None for baseline)
    3. Feed prompt + ground truth tokens, get per-position log probs

    layers: list of layer numbers to collect activations from (e.g. [18] or [9, 18, 27])
    prompt_layer: which layer number to put in the "Layer: X" prefix
    """
    all_log_probs = {d: [] for d in range(1, max_distance + 1)}
    all_top1 = {d: [] for d in range(1, max_distance + 1)}
    all_top5 = {d: [] for d in range(1, max_distance + 1)}
    n = len(token_id_samples)

    if prompt_layer is None:
        prompt_layer = layers[0] if layers else 18

    for i, token_ids in enumerate(token_id_samples):
        if direction == "future":
            max_start = len(token_ids) - max_distance - 1
            min_start = 10
        else:
            min_start = max_distance + 1
            max_start = len(token_ids) - 10

        if max_start <= min_start:
            continue

        pos = random.randint(min_start, max_start)

        if direction == "future":
            ground_truth = token_ids[pos + 1: pos + 1 + max_distance]
        else:
            start = max(0, pos - max_distance)
            ground_truth = token_ids[start:pos][::-1]

        if len(ground_truth) < max_distance:
            continue

        # Collect activations (one per layer)
        activations = None
        if use_activation:
            context_ids = token_ids[:pos + 1]
            input_tensor = torch.tensor([context_ids], device=device)
            attn_mask = torch.ones_like(input_tensor)
            try:
                acts = []
                for layer in layers:
                    act = collect_activation_at_position(
                        model, input_tensor, attn_mask, layer, pos
                    )
                    acts.append(act)
                activations = torch.stack(acts, dim=0)  # [num_layers, d_model]
            except Exception as e:
                print(f"  [{i+1}/{n}] Activation failed: {e}")
                continue

        # Teacher forcing query
        try:
            result = query_ao_teacher_forcing(
                model, tokenizer, activations, ground_truth,
                prompt_layer, direction=direction, device=device,
            )
        except Exception as e:
            print(f"  [{i+1}/{n}] TF query failed: {e}")
            continue

        for d in range(1, max_distance + 1):
            idx = d - 1
            if idx < len(result["log_probs"]):
                all_log_probs[d].append(result["log_probs"][idx])
                all_top1[d].append(result["top1_correct"][idx])
                all_top5[d].append(result["top5_correct"][idx])

        if (i + 1) % 20 == 0 or i == 0:
            t1_1 = sum(all_top1[1]) / max(len(all_top1[1]), 1) * 100
            t1_5 = sum(all_top1[5]) / max(len(all_top1[5]), 1) * 100
            t1_20 = sum(all_top1[20]) / max(len(all_top1[20]), 1) * 100
            print(f"  [{i+1}/{n}] ({label_str}) top1 d=1:{t1_1:.1f}% d=5:{t1_5:.1f}% d=20:{t1_20:.1f}%")

    return all_log_probs, all_top1, all_top5


def aggregate(data_dict):
    """Aggregate list of bools/floats to means."""
    return {d: (sum(v) / len(v) * 100 if v else 0) for d, v in data_dict.items()}


def aggregate_mean(data_dict):
    """Aggregate list of floats to means (for log probs)."""
    return {d: (sum(v) / len(v) if v else 0) for d, v in data_dict.items()}


def plot_results(all_curves, max_distance, output_path, model_name="Qwen3-8B", layer_pcts=None):
    """
    Plot with 2 rows (FineWeb, CoT) x 3 cols (top1, top5, log_prob).
    Each subplot shows baseline, AO 50%, AO 25+50+75%.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distances = list(range(1, max_distance + 1))

    sources = []
    if any("FineWeb" in k for k in all_curves):
        sources.append("FineWeb")
    if any("CoT" in k for k in all_curves):
        sources.append("CoT")

    metrics = [
        ("top1", "Top-1 Accuracy (%)"),
        ("top5", "Top-5 Accuracy (%)"),
        ("log_prob", "Mean Log Probability"),
    ]

    cond_styles = {
        "baseline": {"color": "#9E9E9E", "linestyle": "--", "linewidth": 2},
        "AO 50%": {"color": "#4CAF50", "linestyle": "-", "linewidth": 2},
        "AO 25+50+75%": {"color": "#E91E63", "linestyle": "-", "linewidth": 2.5},
    }

    n_rows = len(sources)
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 6 * n_rows), squeeze=False)

    for row, source in enumerate(sources):
        for col, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row, col]

            for cond_name, style in cond_styles.items():
                key = f"{source} {cond_name}"
                if key not in all_curves:
                    continue
                vals = [all_curves[key][metric_key].get(d, 0) for d in distances]
                ax.plot(distances, vals, label=cond_name, **style)

            ax.set_xlabel("Prediction Distance (tokens)", fontsize=11)
            ax.set_ylabel(metric_label, fontsize=11)
            ax.set_title(f"{source} — {metric_label}", fontsize=12)
            ax.legend(fontsize=10, loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max_distance + 1)

    fig.suptitle(f"AO Prediction Accuracy vs Distance (Teacher Forcing)\n{model_name} — baseline vs 50% vs 25+50+75%",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-distance", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/ao_distance_tf")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    layer_25 = layer_percent_to_layer(args.model, 25)
    layer_50 = layer_percent_to_layer(args.model, 50)
    layer_75 = layer_percent_to_layer(args.model, 75)
    print(f"Model: {args.model}")
    print(f"Layers: 25%={layer_25}, 50%={layer_50}, 75%={layer_75}")
    print(f"Max distance: {args.max_distance}, Samples: {args.n_samples}")
    print(f"Conditions: baseline, AO 50% only, AO 25%+50%+75%")

    model, tokenizer = load_model(args.model, args.device)

    fineweb_samples = load_fineweb_samples(tokenizer, args.n_samples, min_length=args.max_distance + 50)
    cot_samples = None
    if args.corpus and Path(args.corpus).exists():
        cot_samples = load_cot_samples(args.corpus, tokenizer, args.n_samples, min_length=args.max_distance + 50)

    all_curves = {}

    # Define conditions: (label, layers_to_collect, use_activation, prompt_layer)
    conditions = [
        ("baseline", [layer_50], False, layer_50),
        ("AO 50%", [layer_50], True, layer_50),
        ("AO 25+50+75%", [layer_25, layer_50, layer_75], True, layer_50),
    ]

    for source_name, samples in [("FineWeb", fineweb_samples), ("CoT", cot_samples)]:
        if samples is None:
            continue
        for cond_label, cond_layers, use_act, prompt_layer in conditions:
            key = f"{source_name} {cond_label}"
            n_acts = len(cond_layers) if use_act else 0
            print(f"\n{'='*60}")
            print(f"{key} ({n_acts} activations)")
            print(f"{'='*60}")

            lp, t1, t5 = run_tf_experiment(
                model, tokenizer, samples, args.max_distance,
                layers=cond_layers, direction="future",
                use_activation=use_act, prompt_layer=prompt_layer,
                label_str=cond_label, device=args.device,
            )
            all_curves[key] = {
                "top1": aggregate(t1), "top5": aggregate(t5), "log_prob": aggregate_mean(lp),
            }

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "accuracy_vs_distance.json"
    with open(json_path, "w") as f:
        json.dump({
            "model": args.model,
            "layers": {"25%": layer_25, "50%": layer_50, "75%": layer_75},
            "max_distance": args.max_distance, "n_samples": args.n_samples,
            "curves": {k: {m: {str(d): v for d, v in vals.items()} for m, vals in v.items()}
                       for k, v in all_curves.items()},
        }, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Plot
    try:
        plot_results(all_curves, args.max_distance,
                     output_dir / "accuracy_vs_distance.png",
                     model_name=args.model.split("/")[-1], layer_pcts=args.layer_percents)
    except Exception as e:
        print(f"Plotting failed: {e}")

    # Summary table
    print(f"\n{'Distance':>8}", end="")
    for label in all_curves:
        print(f" {label[:18]:>18}", end="")
    print()
    print("-" * (8 + 19 * len(all_curves)))
    for d in [1, 2, 3, 5, 10, 15, 20, 30, 50]:
        if d <= args.max_distance:
            print(f"{d:>8}", end="")
            for label in all_curves:
                val = all_curves[label]["top1"].get(d, 0)
                print(f" {val:>17.1f}%", end="")
            print()


if __name__ == "__main__":
    main()

"""
AO accuracy vs prediction distance.

How far can the oracle predict from a single activation?
Takes activation at position P, asks AO to predict next K tokens,
checks per-token accuracy at each distance d=1..K.

Two data sources:
  - FineWeb (in-distribution for AO pretraining)
  - CoT corpus (OOD — reasoning traces from Qwen3-8B)

Usage:
    python3 src/ao_distance_accuracy.py \
        --model Qwen/Qwen3-8B \
        --max-distance 50 \
        --n-samples 100

Outputs:
    results/ao_distance/accuracy_vs_distance.json
    results/ao_distance/accuracy_vs_distance.png
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
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


def collect_activation_at_position(model, input_ids, attn_mask, layer, position):
    """Get activation at a single position. Returns [d_model]."""
    activation = None
    # Navigate PEFT wrapping to find layer
    for path_fn in [
        lambda: model.base_model.model.model.layers[layer],
        lambda: model.model.model.layers[layer],
        lambda: model.model.layers[layer],
    ]:
        try:
            submodule = path_fn()
            break
        except (AttributeError, IndexError):
            continue
    else:
        raise ValueError(f"Cannot find layer {layer}")

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

    return activation[0, position, :].detach()  # [d_model]


# ── Oracle Query (generation mode) ────────────────────────

def query_ao_generate(model, tokenizer, activation, k_tokens, layer, direction="future", device="cuda"):
    """
    Ask AO to predict next/prev k_tokens from a single activation.
    Returns list of generated token IDs.
    """
    dtype = torch.bfloat16

    if direction == "future":
        prompt_text = f"Can you predict the next {k_tokens} tokens that come after this?"
    else:
        prompt_text = f"Can you predict the previous {k_tokens} tokens that came before this?"

    prefix = f"Layer: {layer}\n{SPECIAL_TOKEN} \n"
    full_prompt = prefix + prompt_text

    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)

    # Find placeholder position
    special_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
    assert len(special_id) == 1
    special_id = special_id[0]

    placeholder_pos = None
    for i, tid in enumerate(input_ids):
        if tid == special_id:
            placeholder_pos = i
            break
    assert placeholder_pos is not None, "Could not find placeholder token"

    input_tensor = torch.tensor([input_ids], device=device)
    attn_mask = torch.ones_like(input_tensor)

    # Steering hook: norm-matched addition at layer 1
    normed = torch.nn.functional.normalize(activation.unsqueeze(0), dim=-1).detach()  # [1, D]

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

        orig = resid[0, placeholder_pos, :]  # [D]
        norm_val = orig.norm()
        steered = (normed[0].to(device, dtype) * norm_val).detach()
        resid[0, placeholder_pos, :] = steered + orig

        return (resid, *rest) if is_tuple else resid

    # Find injection layer (layer 1)
    for path_fn in [
        lambda: model.base_model.model.model.layers[1],
        lambda: model.model.model.layers[1],
        lambda: model.model.layers[1],
    ]:
        try:
            inject_module = path_fn()
            break
        except (AttributeError, IndexError):
            continue

    handle = inject_module.register_forward_hook(steering_hook)
    try:
        with torch.no_grad():
            output = model.generate(
                input_ids=input_tensor,
                attention_mask=attn_mask,
                max_new_tokens=k_tokens + 20,  # some slack
                do_sample=False,
            )
    finally:
        handle.remove()

    generated_ids = output[0][len(input_ids):].tolist()
    return generated_ids


# ── Data Loading ───────────────────────────────────────────

def load_fineweb_samples(tokenizer, n_samples, min_length=200, seed=42):
    """Load FineWeb text samples. Each sample = token IDs of length >= min_length."""
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
        if len(samples) >= n_samples * 3:  # collect extra, subsample later
            break

    rng.shuffle(samples)
    samples = samples[:n_samples]
    print(f"  Got {len(samples)} FineWeb samples (min {min_length} tokens)")
    return samples


def load_cot_samples(corpus_path, tokenizer, n_samples, min_length=200, seed=42):
    """Load CoT corpus samples."""
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
        # Format as the model would see it during generation
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

def run_distance_experiment(model, tokenizer, token_id_samples, max_distance, layer,
                            direction="future", device="cuda"):
    """
    For each sample, pick a random position P, collect activation at P,
    ask AO to predict next/prev max_distance tokens, compare per-token.

    Returns dict mapping distance -> list of bools (correct/not).
    """
    results = {d: [] for d in range(1, max_distance + 1)}
    n = len(token_id_samples)

    for i, token_ids in enumerate(token_id_samples):
        if direction == "future":
            max_start = len(token_ids) - max_distance - 1
            min_start = 10
        else:  # past
            min_start = max_distance + 1
            max_start = len(token_ids) - 10

        if max_start <= min_start:
            print(f"  [{i+1}/{n}] Skipping (too short: {len(token_ids)} tokens)")
            continue

        pos = random.randint(min_start, max_start)

        if direction == "future":
            actual_tokens = token_ids[pos + 1: pos + 1 + max_distance]
        else:  # past
            # Tokens BEFORE pos, in reverse order (d=1 = immediately before)
            start = max(0, pos - max_distance)
            actual_tokens = token_ids[start:pos][::-1]  # reverse so d=1 = closest

        # Build input for activation collection (full context up to pos)
        context_ids = token_ids[:pos + 1]
        input_tensor = torch.tensor([context_ids], device=device)
        attn_mask = torch.ones_like(input_tensor)

        try:
            activation = collect_activation_at_position(
                model, input_tensor, attn_mask, layer, pos
            )
        except Exception as e:
            print(f"  [{i+1}/{n}] Activation collection failed: {e}")
            continue

        try:
            generated_ids = query_ao_generate(
                model, tokenizer, activation, max_distance, layer,
                direction=direction, device=device,
            )
        except Exception as e:
            print(f"  [{i+1}/{n}] Generation failed: {e}")
            continue

        # Compare per-token
        for d in range(1, max_distance + 1):
            if d - 1 < len(generated_ids) and d - 1 < len(actual_tokens):
                correct = (generated_ids[d - 1] == actual_tokens[d - 1])
                results[d].append(correct)

        if (i + 1) % 10 == 0 or i == 0:
            acc_1 = sum(results[1]) / max(len(results[1]), 1) * 100
            acc_5 = sum(results[5]) / max(len(results[5]), 1) * 100 if len(results[5]) > 0 else 0
            acc_20 = sum(results[20]) / max(len(results[20]), 1) * 100 if len(results[20]) > 0 else 0
            print(f"  [{i+1}/{n}] d=1: {acc_1:.1f}%, d=5: {acc_5:.1f}%, d=20: {acc_20:.1f}%")

    return results


def compute_accuracies(results):
    """Convert results dict to accuracy dict."""
    accuracies = {}
    for d, bools in results.items():
        if bools:
            accuracies[d] = sum(bools) / len(bools) * 100
        else:
            accuracies[d] = 0.0
    return accuracies


def plot_results(all_results, max_distance, output_path, model_name="Qwen3-8B", layer_pct=50):
    """Plot accuracy vs distance. all_results is dict of {direction: {source: accuracies}}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directions = list(all_results.keys())
    n_plots = len(directions)
    fig, axes = plt.subplots(1, n_plots, figsize=(12 * n_plots, 6), squeeze=False)

    distances = list(range(1, max_distance + 1))

    for col, direction in enumerate(directions):
        ax = axes[0, col]
        source_data = all_results[direction]

        for source_name, acc_dict in source_data.items():
            vals = [acc_dict.get(d, 0) for d in distances]
            style = "b-o" if "FineWeb" in source_name else "r-s"
            ax.plot(distances, vals, style, markersize=3, label=source_name, linewidth=2)

        dir_label = "Future (predict after)" if direction == "future" else "Past (predict before)"
        ax.set_xlabel("Prediction Distance (tokens)", fontsize=13)
        ax.set_ylabel("Token-level Accuracy (%)", fontsize=13)
        ax.set_title(f"AO {dir_label}\n(Single activation, {model_name}, layer {layer_pct}%)", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_distance + 1)
        all_vals = [v for acc in source_data.values() for v in acc.values()]
        ax.set_ylim(0, max(all_vals, default=1) * 1.15 + 1)
        ax.axvline(x=20, color="gray", linestyle="--", alpha=0.5, label="Training max (standard)")
        ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-distance", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Samples per data source")
    parser.add_argument("--corpus", default=None,
                        help="Path to CoT corpus JSONL (optional, skips CoT if not provided)")
    parser.add_argument("--layer-percent", type=int, default=50)
    parser.add_argument("--directions", nargs="+", default=["future", "past"],
                        choices=["future", "past"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/ao_distance")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    layer = layer_percent_to_layer(args.model, args.layer_percent)
    print(f"Model: {args.model}, Layer: {layer} ({args.layer_percent}%)")
    print(f"Max distance: {args.max_distance}, Samples per source: {args.n_samples}")
    print(f"Directions: {args.directions}")

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Load data sources
    fineweb_samples = load_fineweb_samples(tokenizer, args.n_samples, min_length=args.max_distance + 50)

    cot_samples = None
    if args.corpus and Path(args.corpus).exists():
        cot_samples = load_cot_samples(args.corpus, tokenizer, args.n_samples, min_length=args.max_distance + 50)

    # Run experiments for each direction
    all_results = {}  # {direction: {source_name: accuracies}}
    all_raw = {}

    for direction in args.directions:
        dir_label = "Future" if direction == "future" else "Past"
        all_results[direction] = {}
        all_raw[direction] = {}

        print(f"\n{'='*60}")
        print(f"FineWeb (in-distribution) — {dir_label}")
        print(f"{'='*60}")
        fw_results = run_distance_experiment(
            model, tokenizer, fineweb_samples, args.max_distance, layer,
            direction=direction, device=args.device,
        )
        all_results[direction]["FineWeb (in-distribution)"] = compute_accuracies(fw_results)
        all_raw[direction] = {"fineweb": fw_results}

        if cot_samples:
            print(f"\n{'='*60}")
            print(f"CoT corpus (OOD) — {dir_label}")
            print(f"{'='*60}")
            cot_results = run_distance_experiment(
                model, tokenizer, cot_samples, args.max_distance, layer,
                direction=direction, device=args.device,
            )
            all_results[direction]["CoT corpus (OOD)"] = compute_accuracies(cot_results)
            all_raw[direction]["cot"] = cot_results

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "model": args.model,
        "layer": layer,
        "layer_percent": args.layer_percent,
        "max_distance": args.max_distance,
        "n_samples_per_source": args.n_samples,
        "directions": args.directions,
    }
    for direction in args.directions:
        for source_name, acc in all_results[direction].items():
            key = f"{direction}_{source_name.split()[0].lower()}_accuracy"
            results_data[key] = acc

    json_path = output_dir / "accuracy_vs_distance.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Plot
    try:
        plot_results(all_results, args.max_distance, output_dir / "accuracy_vs_distance.png",
                     model_name=args.model.split("/")[-1], layer_pct=args.layer_percent)
    except Exception as e:
        print(f"Plotting failed: {e} (results still saved as JSON)")

    # Print summary tables
    for direction in args.directions:
        print(f"\n--- {direction.upper()} ---")
        sources = list(all_results[direction].keys())
        header = f"{'Distance':>10}" + "".join(f" {s[:12]:>12}" for s in sources)
        print(header)
        print("-" * len(header))
        for d in [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]:
            if d <= args.max_distance:
                vals = "".join(f" {all_results[direction][s].get(d, 0):>11.1f}%" for s in sources)
                print(f"{d:>10}{vals}")


if __name__ == "__main__":
    main()

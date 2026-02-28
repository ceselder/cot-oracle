#!/usr/bin/env python
"""Label SAE features with LLM using max-activating examples.

Batches multiple features per prompt for speed. For each alive feature, formats
top-N activating examples with highlighted peak token, sends to Gemini Flash Lite
via OpenRouter, and collects short labels.

Usage:
    python scripts/sae_label_features.py \
        --input-dir $CACHE_DIR/sae_features/ \
        --trainer 2 --min-examples 10 --n-examples 10 \
        --model google/gemini-2.5-flash-lite --batch-size 5
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path.home() / ".env")

MODEL_NAME = "Qwen/Qwen3-8B"
SAE_LAYERS = [9, 18, 27]


def decode_context_highlighted(tokenizer, context_tokens: torch.Tensor) -> str:
    """Decode context tokens with the center token highlighted as >>>token<<<."""
    ids = context_tokens.tolist()
    half_w = len(ids) // 2
    parts = []
    for i, tid in enumerate(ids):
        if tid == tokenizer.pad_token_id:
            continue
        text = tokenizer.decode([tid])
        if i == half_w:
            text = f">>>{text}<<<"
        parts.append(text)
    return "".join(parts)


def prepare_feature_examples(tokenizer, top_values, top_contexts, feat_idx, n_examples):
    """Prepare decoded examples for a single feature."""
    vals = top_values[feat_idx]     # [K]
    ctxs = top_contexts[feat_idx]   # [K, W]
    n_valid = (vals > -float("inf")).sum().item()
    n_use = min(n_examples, n_valid)
    examples = []
    for k in range(n_use):
        text = decode_context_highlighted(tokenizer, ctxs[k])
        examples.append((vals[k].item(), text))
    return examples


def format_batch_prompt(feature_batch: list[tuple[int, list[tuple[float, str]]]],
                        layer: int) -> str:
    """Build a prompt that labels multiple features at once."""
    lines = [
        f"Below are top activating examples for {len(feature_batch)} SAE features "
        f"from layer {layer} of Qwen3-8B.",
        "The max-activating token in each example is marked with >>> <<<.",
        "",
    ]
    for feat_idx, examples in feature_batch:
        lines.append(f"--- Feature {feat_idx} ---")
        for i, (act_val, text) in enumerate(examples, 1):
            lines.append(f"Example {i} (activation: {act_val:.2f}):")
            lines.append(text)
        lines.append("")

    feat_ids = [str(f) for f, _ in feature_batch]
    lines.append(
        "For each feature above, provide a JSON object. "
        "Respond with ONLY a JSON array (no markdown fences), one object per feature in order:\n"
        f'[{{"feature": {feat_ids[0]}, "label": "3-8 word description", '
        f'"explanation": "one sentence", "confidence": "low/medium/high"}}, ...]'
    )
    return "\n".join(lines)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
async def call_llm(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def parse_batch_response(text: str, expected_features: list[int]) -> dict[int, dict]:
    """Parse a JSON array response into per-feature labels."""
    # Find JSON array in response
    match = re.search(r"\[[\s\S]*\]", text)
    arr = json.loads(match.group())
    results = {}
    for item in arr:
        # Match by feature field or by position
        feat_id = item.get("feature")
        if feat_id is not None:
            results[int(feat_id)] = {
                "label": item["label"],
                "explanation": item["explanation"],
                "confidence": item["confidence"],
            }
    # Fallback: if no feature fields, assign by position
    if not results and len(arr) == len(expected_features):
        for feat_idx, item in zip(expected_features, arr):
            results[feat_idx] = {
                "label": item["label"],
                "explanation": item["explanation"],
                "confidence": item["confidence"],
            }
    return results


async def label_batch(client: AsyncOpenAI, semaphore: asyncio.Semaphore,
                      model: str, feature_batch: list[tuple[int, list[tuple[float, str]]]],
                      layer: int, log_dir: Path) -> dict[int, dict]:
    """Label a batch of features in one API call."""
    async with semaphore:
        prompt = format_batch_prompt(feature_batch, layer)
        feat_ids = [f for f, _ in feature_batch]
        max_tokens = 128 * len(feature_batch)

        # Log prompt
        batch_name = f"batch_{feat_ids[0]:06d}-{feat_ids[-1]:06d}"
        log_path = log_dir / f"{batch_name}.txt"
        log_path.write_text(prompt)

        text = await call_llm(client, model, prompt, max_tokens)

        with open(log_path, "a") as f:
            f.write(f"\n\n--- RESPONSE ---\n{text}")

        return parse_batch_response(text, feat_ids)


async def label_layer(client: AsyncOpenAI, model: str, tokenizer,
                      tracker_data: dict, layer: int, n_examples: int,
                      min_examples: int, log_dir: Path, output_path: Path,
                      max_concurrent: int, batch_size: int):
    """Label all alive features for one layer."""
    semaphore = asyncio.Semaphore(max_concurrent)

    top_values = tracker_data["top_values"].float()   # [F, K]
    top_contexts = tracker_data["top_contexts"]        # [F, K, W]
    alive_count = tracker_data["alive_count"]          # [F]
    max_activation = tracker_data["max_activation"].float()

    # Resume from existing labels
    labels: dict[str, dict] = {}
    if output_path.exists():
        labels = json.loads(output_path.read_text())
        print(f"  Resuming: {len(labels)} existing labels")

    alive_features = (alive_count >= min_examples).nonzero(as_tuple=True)[0].tolist()
    to_label = [f for f in alive_features if str(f) not in labels]
    print(f"  Layer {layer}: {len(alive_features)} alive (>={min_examples}), "
          f"{len(to_label)} remaining")

    if not to_label:
        return labels

    layer_log_dir = log_dir / f"layer{layer}"
    layer_log_dir.mkdir(parents=True, exist_ok=True)

    # Prepare all feature examples upfront
    print(f"  Preparing examples...")
    feature_examples = []
    for feat_idx in tqdm(to_label, desc=f"  Decoding L{layer}", leave=False):
        examples = prepare_feature_examples(tokenizer, top_values, top_contexts, feat_idx, n_examples)
        feature_examples.append((feat_idx, examples))

    # Group into LLM batches
    llm_batches = [feature_examples[i:i + batch_size]
                   for i in range(0, len(feature_examples), batch_size)]

    # Process in save-interval chunks (save every 200 LLM calls)
    save_interval = 200
    for chunk_start in tqdm(range(0, len(llm_batches), save_interval),
                            desc=f"Layer {layer}",
                            total=(len(llm_batches) + save_interval - 1) // save_interval):
        chunk = llm_batches[chunk_start : chunk_start + save_interval]

        tasks = [label_batch(client, semaphore, model, batch, layer, layer_log_dir)
                 for batch in chunk]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for batch, result in zip(chunk, results):
            feat_ids = [f for f, _ in batch]
            if isinstance(result, Exception):
                print(f"  Batch {feat_ids[0]}-{feat_ids[-1]}: ERROR - {result}")
                for feat_idx in feat_ids:
                    labels[str(feat_idx)] = {"label": "ERROR", "error": str(result)}
            else:
                for feat_idx in feat_ids:
                    if feat_idx in result:
                        entry = result[feat_idx]
                        entry["alive_count"] = int(alive_count[feat_idx].item())
                        entry["max_activation"] = float(max_activation[feat_idx].item())
                        labels[str(feat_idx)] = entry
                    else:
                        labels[str(feat_idx)] = {"label": "ERROR", "error": "missing from batch response"}

        output_path.write_text(json.dumps(labels, indent=2))

    print(f"  Saved {len(labels)} labels -> {output_path}")
    return labels


async def async_main(args):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    input_dir = Path(os.path.expandvars(args.input_dir)) / f"trainer_{args.trainer}"
    if args.output_dir:
        output_dir = Path(os.path.expandvars(args.output_dir))
    else:
        output_dir = input_dir / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "labeling_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    layers = args.layers or SAE_LAYERS
    for layer in layers:
        pt_path = input_dir / f"topk_layer{layer}.pt"
        print(f"\nLoading {pt_path}...")
        tracker_data = torch.load(pt_path, map_location="cpu", weights_only=True)

        output_path = output_dir / f"labels_layer{layer}_trainer{args.trainer}.json"

        await label_layer(
            client, args.model, tokenizer, tracker_data, layer,
            args.n_examples, args.min_examples, log_dir, output_path,
            args.max_concurrent, args.batch_size,
        )

    # Print summary
    print("\n=== Summary ===")
    for layer in layers:
        output_path = output_dir / f"labels_layer{layer}_trainer{args.trainer}.json"
        if output_path.exists():
            data = json.loads(output_path.read_text())
            n_total = len(data)
            n_high = sum(1 for v in data.values() if v.get("confidence") == "high")
            n_err = sum(1 for v in data.values() if v.get("label") == "ERROR")
            print(f"Layer {layer}: {n_total} labeled, {n_high} high-confidence, {n_err} errors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=str, required=True, help="Dir with topk_layer*.pt files")
    parser.add_argument("--trainer", type=int, default=2)
    parser.add_argument("--min-examples", type=int, default=10, help="Min alive count to label a feature")
    parser.add_argument("--n-examples", type=int, default=10, help="Top-N examples per prompt")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash-lite")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5, help="Features per LLM call")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Layers to label (default: all)")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

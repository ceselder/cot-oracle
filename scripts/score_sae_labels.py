#!/usr/bin/env python
"""Score SAE feature autolabel quality via detection task.

For each sampled feature, the LLM is shown a mix of held-out real activating
contexts and random non-activating contexts, and must identify which are positive
based solely on the feature's label + explanation.

Detection accuracy (chance = 50%) measures how well the label predicts what
activates the feature on held-out examples not used during labeling.

Methodology (standard SAE autointerp benchmark):
  - Positives: examples 10-29 from top_contexts (held out from labeling, which used 0-9)
  - Negatives (default): random top-activating contexts from OTHER features (CoT corpus)
  - Negatives (--fineweb-negatives): random passages streamed from FineWeb
  - Present N_POS + N_NEG shuffled contexts, ask LLM for positive IDs
  - Compute accuracy, precision, recall per feature
  - Aggregate by layer and confidence tier

Usage:
    python scripts/score_sae_labels.py --trainer 2 --n-per-layer 200 --n-pos 5 --n-neg 5
    python scripts/score_sae_labels.py --trainer 2 --n-per-layer 200 --fineweb-negatives
    python scripts/score_sae_labels.py --trainer 2 --n-per-layer 50  # quick test

Output:
    $CACHE_DIR/sae_features/trainer_2/trainer_2/detection_scores_trainer2[_fineweb].json
    $CACHE_DIR/sae_features/trainer_2/trainer_2/detection_logs[_fineweb]/
"""

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path.home() / ".env")

DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_FINEWEB_REPO = "HuggingFaceFW/fineweb"
N_LABEL_EXAMPLES = 10  # how many examples were used during labeling (used examples 0..N-1)


def build_fineweb_neg_pool(tokenizer, n_needed: int, context_window: int,
                           seed: int, fineweb_repo: str = DEFAULT_FINEWEB_REPO) -> list[str]:
    """Stream FineWeb, extract random token windows of size context_window as negatives."""
    from datasets import load_dataset
    rng = random.Random(seed)
    half_w = context_window // 2
    pool: list[str] = []
    ds = load_dataset(fineweb_repo, split="train", streaming=True)
    for row in tqdm(ds, desc="Streaming FineWeb negatives", total=n_needed):
        text = row["text"]
        if len(text) < 50:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < context_window:
            continue
        # Pick a random center position
        center = rng.randint(half_w, len(ids) - half_w - 1)
        window = ids[center - half_w: center + half_w + 1]
        parts = []
        for i, tid in enumerate(window):
            tok = tokenizer.decode([tid])
            parts.append(f">>>{tok}<<<" if i == half_w else tok)
        pool.append("".join(parts))
        if len(pool) >= n_needed:
            break
    return pool


def default_layers(model_name: str) -> list[int]:
    if "Qwen3-8B" in model_name:
        return [9, 18, 27]
    if "Qwen3-0.6B" in model_name or "Qwen3-1.7B" in model_name:
        return [7, 14, 21]
    raise ValueError(f"Specify --layers for unsupported model: {model_name}")


def decode_context(tokenizer, context_tokens: torch.Tensor, highlight_center: bool = True) -> str:
    """Decode context tokens, optionally highlighting the center (peak) token."""
    ids = context_tokens.tolist()
    half_w = len(ids) // 2
    parts = []
    for i, tid in enumerate(ids):
        if tid == tokenizer.pad_token_id:
            continue
        text = tokenizer.decode([tid])
        if highlight_center and i == half_w:
            text = f">>>{text}<<<"
        parts.append(text)
    return "".join(parts)


def sample_features(
    labels: dict[str, dict],
    alive_count: torch.Tensor,
    n_per_layer: int,
    n_held_out_min: int,
    rng: random.Random,
) -> list[int]:
    """Sample features stratified by confidence, requiring enough held-out examples."""
    valid = [
        int(fid) for fid, v in labels.items()
        if v.get("label") not in ("ERROR", None)
        and alive_count[int(fid)].item() > N_LABEL_EXAMPLES + n_held_out_min
    ]

    by_conf = {"high": [], "medium": [], "low": []}
    for fid in valid:
        conf = labels[str(fid)].get("confidence", "medium")
        by_conf.get(conf, by_conf["medium"]).append(fid)

    # Proportional allocation across confidence tiers
    n_high = min(len(by_conf["high"]), n_per_layer * 6 // 10)
    n_med  = min(len(by_conf["medium"]), n_per_layer * 3 // 10)
    n_low  = min(len(by_conf["low"]),  n_per_layer - n_high - n_med)
    n_low  = min(n_low, n_per_layer - n_high - n_med)

    rng.shuffle(by_conf["high"])
    rng.shuffle(by_conf["medium"])
    rng.shuffle(by_conf["low"])

    sampled = by_conf["high"][:n_high] + by_conf["medium"][:n_med] + by_conf["low"][:n_low]
    rng.shuffle(sampled)
    return sampled[:n_per_layer]


def build_detection_prompt(
    label: str,
    explanation: str,
    contexts: list[tuple[int, str]],  # (context_id, decoded_text)
) -> str:
    lines = [
        "You are evaluating whether a feature label accurately describes what activates an SAE feature.",
        "",
        f'Feature label: "{label}"',
        f'Explanation: "{explanation}"',
        "",
        f"Below are {len(contexts)} text contexts. The max-activating token is marked with >>> <<<.",
        "Some contexts strongly activate this feature; others do not.",
        "",
    ]
    for ctx_id, text in contexts:
        lines.append(f"[{ctx_id}] {text}")
    lines.append("")
    lines.append(
        "Which context IDs correspond to examples that would strongly activate this feature? "
        "Return ONLY a JSON array of integer IDs, e.g. [1, 3, 5]. No other text."
    )
    return "\n".join(lines)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
async def call_llm(client: AsyncOpenAI, model: str, prompt: str) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=128,
    )
    return resp.choices[0].message.content.strip()


def parse_id_list(text: str) -> list[int]:
    """Parse a JSON array of ints from LLM response."""
    import re
    match = re.search(r"\[[\d,\s]*\]", text)
    if not match:
        return []
    return [int(x) for x in json.loads(match.group())]


def compute_scores(predicted_ids: list[int], positive_ids: set[int], all_ids: list[int]) -> dict:
    predicted = set(predicted_ids) & set(all_ids)
    tp = len(predicted & positive_ids)
    fp = len(predicted - positive_ids)
    fn = len(positive_ids - predicted)
    tn = len(set(all_ids) - positive_ids - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy  = (tp + tn) / len(all_ids) if all_ids else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


async def score_feature(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    feat_idx: int,
    label_entry: dict,
    tokenizer,
    top_contexts: torch.Tensor,  # [K, W]
    top_values: torch.Tensor,    # [K]
    neg_contexts: list[str],     # pre-decoded negatives
    n_pos: int,
    n_neg: int,
    log_dir: Path,
    rng: random.Random,
) -> dict:
    label = label_entry["label"]
    explanation = label_entry.get("explanation", "")

    # Held-out positives: examples N_LABEL_EXAMPLES .. N_LABEL_EXAMPLES+n_pos-1
    pos_decoded = [
        decode_context(tokenizer, top_contexts[N_LABEL_EXAMPLES + i])
        for i in range(n_pos)
    ]

    # Assign context IDs and shuffle
    all_texts = pos_decoded + neg_contexts[:n_neg]
    ids = list(range(len(all_texts)))
    rng.shuffle(ids)
    shuffled = [(ids[i], all_texts[i]) for i in range(len(all_texts))]
    positive_ids = set(ids[:n_pos])

    prompt = build_detection_prompt(label, explanation, shuffled)

    # Log prompt
    log_path = log_dir / f"feat_{feat_idx:06d}.txt"
    log_path.write_text(prompt)

    async with semaphore:
        try:
            response_text = await call_llm(client, model, prompt)
        except Exception as e:
            log_path.write_text(prompt + f"\n\n--- ERROR ---\n{e}")
            return {"feat_idx": feat_idx, "error": str(e), "confidence": label_entry.get("confidence")}

    log_path.write_text(prompt + f"\n\n--- RESPONSE ---\n{response_text}")

    predicted = parse_id_list(response_text)
    all_ids = [ctx_id for ctx_id, _ in shuffled]
    scores = compute_scores(predicted, positive_ids, all_ids)

    return {
        "feat_idx": feat_idx,
        "label": label,
        "confidence": label_entry.get("confidence", "medium"),
        "alive_count": label_entry.get("alive_count"),
        "max_activation": label_entry.get("max_activation"),
        **scores,
    }


async def score_layer(
    client: AsyncOpenAI,
    model: str,
    tokenizer,
    tracker_data: dict,
    labels: dict[str, dict],
    layer: int,
    n_per_layer: int,
    n_pos: int,
    n_neg: int,
    max_concurrent: int,
    log_dir: Path,
    output_path: Path,
    seed: int,
    external_neg_pool: list[str] | None = None,
) -> list[dict]:
    rng = random.Random(seed + layer)

    top_values   = tracker_data["top_values"].float()    # [F, K]
    top_contexts = tracker_data["top_contexts"]           # [F, K, W]
    alive_count  = tracker_data["alive_count"]            # [F]

    # Resume
    results: list[dict] = []
    done_feats: set[int] = set()
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        results = [r for r in existing if r.get("layer") == layer]
        done_feats = {r["feat_idx"] for r in results if "error" not in r}
        print(f"  Resuming layer {layer}: {len(done_feats)} already scored")

    feature_ids = sample_features(labels, alive_count, n_per_layer, n_pos, rng)
    feature_ids = [f for f in feature_ids if f not in done_feats]
    print(f"  Layer {layer}: scoring {len(feature_ids)} features ({n_pos} pos / {n_neg} neg each)")

    if not feature_ids:
        return results

    if external_neg_pool is not None:
        # Use pre-built external negatives (e.g. FineWeb)
        neg_pool = list(external_neg_pool)
        rng.shuffle(neg_pool)
    else:
        # Build negatives by sampling from other features' top contexts (CoT corpus)
        all_feat_indices = list(range(top_contexts.shape[0]))
        neg_pool: list[str] = []
        rng_neg = random.Random(seed + layer + 1000)
        for _ in range(n_neg * len(feature_ids) * 3):
            rand_feat = rng_neg.choice(all_feat_indices)
            rand_ex   = rng_neg.randint(0, top_contexts.shape[1] - 1)
            if alive_count[rand_feat].item() > rand_ex:
                neg_pool.append(decode_context(tokenizer, top_contexts[rand_feat, rand_ex]))
        rng_neg.shuffle(neg_pool)

    semaphore = asyncio.Semaphore(max_concurrent)
    layer_log_dir = log_dir / f"layer{layer}"
    layer_log_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i, feat_idx in enumerate(feature_ids):
        neg_slice = neg_pool[i * n_neg: (i + 1) * n_neg]
        if len(neg_slice) < n_neg:
            continue  # not enough negatives
        tasks.append(score_feature(
            client, semaphore, model, feat_idx,
            labels[str(feat_idx)], tokenizer,
            top_contexts[feat_idx], top_values[feat_idx],
            neg_slice, n_pos, n_neg,
            layer_log_dir, random.Random(seed + feat_idx),
        ))

    save_interval = 50
    for chunk_start in tqdm(range(0, len(tasks), save_interval),
                            desc=f"Layer {layer}",
                            total=(len(tasks) + save_interval - 1) // save_interval):
        chunk = tasks[chunk_start: chunk_start + save_interval]
        chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
        for r in chunk_results:
            if isinstance(r, Exception):
                print(f"  Task error: {r}")
            else:
                r["layer"] = layer
                results.append(r)
        output_path.write_text(json.dumps(results, indent=2))

    return results


def print_summary(all_results: list[dict]) -> None:
    from collections import defaultdict

    print("\n=== Detection Score Summary ===")
    print(f"{'Layer':<8} {'Confidence':<12} {'N':<6} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<6}")
    print("-" * 65)

    by_layer_conf: dict[tuple, list] = defaultdict(list)
    for r in all_results:
        if "error" in r:
            continue
        key = (r.get("layer"), r.get("confidence", "?"))
        by_layer_conf[key].append(r)

    for (layer, conf), items in sorted(by_layer_conf.items()):
        accs = [x["accuracy"] for x in items]
        precs = [x["precision"] for x in items]
        recs = [x["recall"] for x in items]
        f1s = [x["f1"] for x in items]
        n = len(items)
        print(f"{str(layer):<8} {conf:<12} {n:<6} "
              f"{sum(accs)/n:.3f}     {sum(precs)/n:.3f}       {sum(recs)/n:.3f}   {sum(f1s)/n:.3f}")

    # Overall per layer
    print()
    by_layer: dict = defaultdict(list)
    for r in all_results:
        if "error" not in r:
            by_layer[r.get("layer")].append(r)
    for layer, items in sorted(by_layer.items()):
        accs = [x["accuracy"] for x in items]
        n = len(items)
        print(f"Layer {layer} overall: n={n}, mean_accuracy={sum(accs)/n:.3f}  (chance=0.50)")


async def async_main(args):
    cache_dir = Path(os.environ["CACHE_DIR"])
    input_dir = cache_dir / "sae_features" / f"trainer_{args.trainer}" / f"trainer_{args.trainer}"
    output_dir = input_dir

    suffix = "_fineweb" if args.fineweb_negatives else ""
    log_dir = output_dir / f"detection_logs{suffix}"
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"detection_scores_trainer{args.trainer}{suffix}.json"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    layers = args.layers or default_layers(args.tokenizer_model)

    # Build FineWeb neg pool once (shared across all layers to avoid re-streaming)
    fineweb_neg_pool: list[str] | None = None
    if args.fineweb_negatives:
        context_window = 41  # same as topk collection
        n_needed = args.n_neg * args.n_per_layer * len(layers) * 2  # 2x buffer
        print(f"Building FineWeb negative pool ({n_needed} passages)...")
        fineweb_neg_pool = build_fineweb_neg_pool(
            tokenizer, n_needed, context_window, args.seed, args.fineweb_repo,
        )
        print(f"  Built {len(fineweb_neg_pool)} FineWeb negatives")

    all_results: list[dict] = []
    if output_path.exists():
        all_results = json.loads(output_path.read_text())

    for layer in layers:
        pt_path = input_dir / f"topk_layer{layer}.pt"
        label_path = input_dir / "labels" / f"labels_layer{layer}_trainer{args.trainer}.json"
        print(f"\nLayer {layer}: loading data...")
        tracker_data = torch.load(pt_path, map_location="cpu", weights_only=True)
        labels = json.loads(label_path.read_text())

        layer_results = await score_layer(
            client, args.model, tokenizer, tracker_data, labels, layer,
            args.n_per_layer, args.n_pos, args.n_neg,
            args.max_concurrent, log_dir, output_path, args.seed,
            external_neg_pool=fineweb_neg_pool,
        )
        all_results = [r for r in all_results if r.get("layer") != layer] + layer_results
        output_path.write_text(json.dumps(all_results, indent=2))

    print_summary(all_results)
    print(f"\nFull results saved to: {output_path}")
    print(f"Per-feature logs: {log_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", type=int, default=2)
    parser.add_argument("--n-per-layer", type=int, default=200, help="Features to score per layer")
    parser.add_argument("--n-pos", type=int, default=5, help="Held-out positive contexts per feature")
    parser.add_argument("--n-neg", type=int, default=5, help="Negative contexts per feature")
    parser.add_argument("--tokenizer-model", type=str, default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fineweb-negatives", action="store_true",
                        help="Use FineWeb text passages as negatives instead of other-feature CoT contexts")
    parser.add_argument("--fineweb-repo", type=str, default=DEFAULT_FINEWEB_REPO)
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

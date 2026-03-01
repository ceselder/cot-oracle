#!/usr/bin/env python3
"""
Probe baselines v2: Eleuther-style attention probes on full CoT token sequences.

Changes from v1:
  - Configurable stride (default 1 = every CoT token)
  - Eleuther's exact attention probe: learned query + ALiBi position bias + multi-head
  - Probes see full token sequence with padding/masking (no mean-pooling)
  - Mean-pool and last-pos linear probes as comparison baselines
  - Two-phase: extract all activations first, then free LLM and train probes
  - Dynamic batch collation (no pre-padded mega-tensors)

Reference: https://blog.eleuther.ai/attention-probes/
           https://github.com/EleutherAI/attention-probes

Usage:
    python scripts/run_probe_baselines.py
    python scripts/run_probe_baselines.py --stride 1 --max-train 10000
    python scripts/run_probe_baselines.py --datasets correctness sycophancy
"""

import argparse
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm, trange

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.ao import EarlyStopException, get_hf_submodule

LAYERS = [9, 18, 27]
SEED = 42

# All cleaned datasets: (hf_repo, task_type, label_field_or_target)
DATASETS = {
    "hint_admission": (
        "mats-10-sprint-cs-jb/cot-oracle-hint-admission-cleaned",
        "binary", "label",
    ),
    "atypical_answer": (
        "mats-10-sprint-cs-jb/cot-oracle-atypical-answer-cleaned",
        "binary", "label",
    ),
    "reasoning_termination": (
        "mats-10-sprint-cs-jb/cot-oracle-reasoning-termination-cleaned",
        "binary", "label",
    ),
    "correctness": (
        "mats-10-sprint-cs-jb/cot-oracle-correctness-cleaned",
        "binary", "label",
    ),
    "decorative_cot": (
        "mats-10-sprint-cs-jb/cot-oracle-decorative-cot-cleaned",
        "binary", "label",
    ),
    "sycophancy": (
        "mats-10-sprint-cs-jb/cot-oracle-sycophancy-cleaned",
        "binary", "label",
    ),
    "truthfulqa_verb": (
        "mats-10-sprint-cs-jb/cot-oracle-eval-hinted-mcq-truthfulqa-verbalized",
        "binary", "label",
    ),
    "truthfulqa_unverb": (
        "mats-10-sprint-cs-jb/cot-oracle-eval-hinted-mcq-truthfulqa-unverbalized",
        "binary", "label",
    ),
    "answer_trajectory": (
        "mats-10-sprint-cs-jb/cot-oracle-answer-trajectory-cleaned",
        "regression", "confidence",
    ),
}

# Binarize hint labels: hint_used_correct + hint_used_wrong → hint_used
HINT_BINARIZE = {
    "hint_used_correct": "hint_used",
    "hint_used_wrong": "hint_used",
    "hint_resisted": "hint_resisted",
}


# ── Eleuther Attention Probe (exact architecture) ──────────────────────────

class AttentionProbe(nn.Module):
    """Eleuther AI's attention probe architecture.

    Learned query projection (init zeros) + ALiBi-style position bias.
    Multi-head: each head independently attends over the sequence, outputs are
    summed across heads and sequence positions.

    Reference: https://github.com/EleutherAI/attention-probes
    """

    def __init__(self, d_in: int, n_heads: int, output_dim: int = 1):
        super().__init__()
        self.q = nn.Linear(d_in, n_heads, bias=False)
        self.q.weight.data.zero_()                          # Eleuther init
        self.v = nn.Linear(d_in, n_heads * output_dim)
        self.n_heads = n_heads
        self.output_dim = output_dim
        self.position_weight = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x, mask, position):
        """
        x:        [B, S, D]  hidden states
        mask:     [B, S]     bool, True = valid
        position: [B, S]     int, position indices
        Returns:  [B, output_dim]
        """
        k = self.q(x)                                       # [B, S, H]
        k = k - ((1 - mask.float()).unsqueeze(-1) * 1e9)    # mask invalid → -inf
        k = k + position.unsqueeze(-1).float() * self.position_weight  # ALiBi
        p = F.softmax(k, dim=-2)                            # [B, S, H]
        v = self.v(x).unflatten(-1, (self.n_heads, self.output_dim))  # [B,S,H,O]
        o = (p.unsqueeze(-1) * v).sum(dim=(-3, -2))         # sum S and H → [B,O]
        return o


# ── Activation Extraction ──────────────────────────────────────────────────

def extract_activations(model, input_ids, attn_mask, positions, layers):
    """Single forward pass → {layer: [K, D]} on CPU float16."""
    submodules = {l: get_hf_submodule(model, l) for l in layers}
    max_layer = max(layers)
    acts = {}
    mod_to_layer = {id(s): l for l, s in submodules.items()}

    def hook_fn(module, _inp, out):
        layer = mod_to_layer[id(module)]
        raw = out[0] if isinstance(out, tuple) else out
        acts[layer] = raw[0, positions, :].detach().cpu().half()
        if layer == max_layer:
            raise EarlyStopException()

    handles = [s.register_forward_hook(hook_fn) for s in submodules.values()]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn_mask)
    except EarlyStopException:
        pass
    finally:
        for h in handles:
            h.remove()
    return acts


def get_cot_positions(prompt_len, total_len, stride=1):
    """Get all CoT token positions at the given stride."""
    all_pos = list(range(prompt_len, total_len, stride))
    # Always include the very last token
    if all_pos and all_pos[-1] != total_len - 1:
        all_pos.append(total_len - 1)
    return all_pos


def process_dataset(ds_name, hf_repo, model, tokenizer, device,
                    max_train, max_test, stride):
    """Load dataset and extract per-token activations (full CoT, no truncation).

    Returns (train_items, test_items) where each item is
    (acts_by_layer: {layer: [n_pos, D]}, row_dict, n_positions).
    """
    print(f"\n  Loading {ds_name} from {hf_repo}...")
    try:
        ds = load_dataset(hf_repo)
    except Exception as e:
        print(f"    FAILED: {e}")
        return None, None

    splits = {}
    for split_name, max_n in [("train", max_train), ("test", max_test)]:
        if split_name not in ds:
            continue
        split = ds[split_name]
        if max_n and len(split) > max_n:
            split = split.shuffle(seed=SEED).select(range(max_n))
        splits[split_name] = split

    if "test" not in splits:
        return None, None

    model.eval()
    results = {}
    for split_name, split in splits.items():
        items = []
        skipped = 0
        pos_counts = []
        for row in tqdm(split, desc=f"    {ds_name}/{split_name}", leave=False):
            cot_text = (row.get("cot_text") or "").strip()
            prompt = row.get("hinted_prompt") or row.get("question") or ""
            if not cot_text or not prompt:
                skipped += 1
                continue

            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            full_text = formatted + cot_text

            prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
            tok_out = tokenizer(full_text, return_tensors="pt",
                                add_special_tokens=False)
            seq_len = tok_out["input_ids"].shape[1]

            positions = get_cot_positions(
                len(prompt_ids), seq_len, stride=stride,
            )
            positions = [p for p in positions if p < seq_len]
            if len(positions) < 2:
                skipped += 1
                continue

            try:
                acts = extract_activations(
                    model,
                    tok_out["input_ids"].to(device),
                    tok_out["attention_mask"].to(device),
                    positions, LAYERS,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped += 1
                continue
            items.append((acts, dict(row), len(positions)))
            pos_counts.append(len(positions))

        results[split_name] = items
        if pos_counts:
            print(f"    {split_name}: {len(items)} ok, {skipped} skipped | "
                  f"positions: mean={sum(pos_counts)/len(pos_counts):.0f} "
                  f"min={min(pos_counts)} max={max(pos_counts)}")

    return results.get("train", []), results.get("test", [])


def binarize_labels(items, ds_name):
    """Merge hint_used_correct/wrong → hint_used for hint-type datasets."""
    if ds_name not in ("hint_admission", "truthfulqa_verb", "truthfulqa_unverb"):
        return items
    return [
        (acts, {**row, "label": HINT_BINARIZE.get(row["label"], row["label"])}, n)
        for acts, row, n in items
    ]


# ── Dynamic Batch Collation ────────────────────────────────────────────────

def collate_batch(items, indices, layers):
    """Pad a mini-batch of items for the given layer(s).

    Returns x [B, S_max, D], mask [B, S_max], position [B, S_max]
    where D = d_model * len(layers).
    """
    batch = [items[i] for i in indices]
    max_seq = max(n for _, _, n in batch)
    d_model = batch[0][0][layers[0]].shape[1]
    n_layers = len(layers)
    B = len(batch)

    x = torch.zeros(B, max_seq, d_model * n_layers)
    mask = torch.zeros(B, max_seq, dtype=torch.bool)
    position = torch.zeros(B, max_seq, dtype=torch.long)

    for i, (acts, _, n_pos) in enumerate(batch):
        for j, l in enumerate(layers):
            x[i, :n_pos, j * d_model:(j + 1) * d_model] = acts[l].float()
        mask[i, :n_pos] = True
        position[i, :n_pos] = torch.arange(n_pos)

    return x, mask, position


# ── Probe Training ─────────────────────────────────────────────────────────

def train_attn_probe(train_items, y_train, test_items, layers,
                     n_classes, n_heads=4, lr=1e-3, epochs=300,
                     batch_size=256, patience=30, device="cuda"):
    """Train Eleuther attention probe with AdamW + early stopping.

    Returns: predictions tensor on test set.
    """
    d_in = train_items[0][0][layers[0]].shape[1] * len(layers)
    is_regression = (n_classes == 0)
    output_dim = 1 if (n_classes <= 2 or is_regression) else n_classes

    probe = AttentionProbe(d_in, n_heads, output_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    N = len(train_items)

    best_loss, best_epoch, best_state = float("inf"), 0, None
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(N).tolist()
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            bx, bm, bp = collate_batch(train_items, idx, layers)
            bx, bm, bp = bx.to(device), bm.to(device), bp.to(device)
            by = y_train[idx].to(device)

            opt.zero_grad(set_to_none=True)
            out = probe(bx, bm, bp)

            if is_regression:
                loss = F.mse_loss(out.squeeze(-1), by.float())
            elif n_classes <= 2:
                loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), by.float())
            else:
                loss = F.cross_entropy(out, by.long())

            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
        elif ep - best_epoch >= patience:
            break

    if best_state:
        probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Evaluate
    probe.eval()
    all_preds = []
    N_te = len(test_items)
    with torch.no_grad():
        for start in range(0, N_te, batch_size):
            idx = list(range(start, min(start + batch_size, N_te)))
            bx, bm, bp = collate_batch(test_items, idx, layers)
            bx, bm, bp = bx.to(device), bm.to(device), bp.to(device)
            out = probe(bx, bm, bp)
            all_preds.append(out.cpu())
    preds = torch.cat(all_preds, dim=0)

    if is_regression:
        return preds.squeeze(-1)
    elif n_classes <= 2:
        return (preds.squeeze(-1) > 0).long()
    else:
        return preds.argmax(dim=-1)


def train_linear_probe(X_tr, y_tr, X_te, n_classes,
                       lr=0.01, epochs=100, wd=1e-4,
                       batch_size=512, device="cuda"):
    """Simple linear probe on pooled features (mean-pool or last-pos)."""
    is_regression = (n_classes == 0)
    output_dim = 1 if (n_classes <= 2 or is_regression) else n_classes

    # Standardize
    mu = X_tr.mean(dim=0, keepdim=True)
    std = X_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    X_tr_s = ((X_tr - mu) / std).to(device)
    X_te_s = ((X_te - mu) / std).to(device)
    y_tr_d = y_tr.to(device)

    probe = nn.Linear(X_tr.shape[1], output_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(len(X_tr_s), device=device)
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad(set_to_none=True)
            out = probe(X_tr_s[idx])
            if is_regression:
                loss = F.mse_loss(out.squeeze(-1), y_tr_d[idx].float())
            elif n_classes <= 2:
                loss = F.binary_cross_entropy_with_logits(
                    out.squeeze(-1), y_tr_d[idx].float())
            else:
                loss = F.cross_entropy(out, y_tr_d[idx].long())
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(X_te_s)
    if is_regression:
        return preds.squeeze(-1).cpu()
    elif n_classes <= 2:
        return (preds.squeeze(-1) > 0).long().cpu()
    else:
        return preds.argmax(dim=-1).cpu()


# ── Pooling helpers ────────────────────────────────────────────────────────

def mean_pool_vec(items, layers):
    """Mean-pool across positions, concat layers → [N, D*n_layers]."""
    return torch.stack([
        torch.cat([acts[l].float().mean(dim=0) for l in layers], dim=-1)
        for acts, _, _ in items
    ])


def last_pool_vec(items, layers):
    """Last position per layer, concat → [N, D*n_layers]."""
    return torch.stack([
        torch.cat([acts[l][-1].float() for l in layers], dim=-1)
        for acts, _, _ in items
    ])


# ── Scoring ────────────────────────────────────────────────────────────────

def balanced_accuracy(preds, gt):
    classes = sorted(set(gt))
    per_class = []
    for c in classes:
        mask = [g == c for g in gt]
        n = sum(mask)
        if n == 0:
            continue
        correct = sum(p == g for p, g, m in zip(preds, gt, mask) if m)
        per_class.append(correct / n)
    return sum(per_class) / len(per_class) if per_class else 0.0


def mae(preds, gt):
    return sum(abs(p - g) for p, g in zip(preds, gt)) / max(len(gt), 1)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Probe baselines v2: Eleuther attention probes")
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--max-test", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1,
                        help="Token stride for extraction (1=every token)")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="data/probe_baseline_results_v2.json")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="Number of attention heads in probe")
    parser.add_argument("--attn-epochs", type=int, default=300)
    parser.add_argument("--attn-lr", type=float, default=1e-3)
    parser.add_argument("--attn-batch-size", type=int, default=256)
    args = parser.parse_args()

    datasets_to_run = args.datasets or list(DATASETS.keys())

    # ── Phase 1: Load model + extract activations ──
    print("Loading Qwen3-8B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    print(f"  Model loaded. Stride={args.stride}, full CoT (no truncation)\n")

    all_data = {}
    for ds_name in datasets_to_run:
        if ds_name not in DATASETS:
            print(f"Unknown dataset: {ds_name}")
            continue
        hf_repo, task_type, target_field = DATASETS[ds_name]
        print(f"\n{'=' * 60}")
        print(f"Extracting: {ds_name}")
        print(f"{'=' * 60}")
        t0 = time.time()
        train_items, test_items = process_dataset(
            ds_name, hf_repo, model, tokenizer, args.device,
            max_train=args.max_train, max_test=args.max_test,
            stride=args.stride,
        )
        if train_items and test_items:
            train_items = binarize_labels(train_items, ds_name)
            test_items = binarize_labels(test_items, ds_name)
            all_data[ds_name] = (train_items, test_items)
            print(f"  Done: {len(train_items)} train, {len(test_items)} test "
                  f"({time.time() - t0:.0f}s)")

    # ── Phase 2: Free model, train probes ──
    print("\n\nFreeing model for probe training...")
    del model
    torch.cuda.empty_cache()
    gc.collect()

    all_results = {}

    for ds_name in datasets_to_run:
        if ds_name not in all_data:
            continue

        hf_repo, task_type, target_field = DATASETS[ds_name]
        train_items, test_items = all_data[ds_name]

        print(f"\n{'=' * 60}")
        print(f"Training probes: {ds_name} ({task_type})")
        print(f"{'=' * 60}")

        ds_results = {"n_train": len(train_items), "n_test": len(test_items)}

        # Build labels
        if task_type in ("binary", "multiclass"):
            all_labels = sorted(set(
                row[target_field] for _, row, _ in train_items + test_items
            ))
            label2idx = {l: i for i, l in enumerate(all_labels)}
            n_classes = len(all_labels)
            y_train = torch.tensor(
                [label2idx[row[target_field]] for _, row, _ in train_items])
            y_test_raw = [row[target_field] for _, row, _ in test_items]

            train_dist = Counter(row[target_field] for _, row, _ in train_items)
            test_dist = Counter(row[target_field] for _, row, _ in test_items)
            print(f"  Classes: {all_labels}")
            print(f"  Train: {dict(train_dist)}")
            print(f"  Test:  {dict(test_dist)}")
        else:
            n_classes = 0
            y_train = torch.tensor(
                [row[target_field] for _, row, _ in train_items]).float()
            y_test_raw = [row[target_field] for _, row, _ in test_items]
            y_test_t = torch.tensor(y_test_raw).float()
            print(f"  Train: mean={y_train.mean():.3f} std={y_train.std():.3f}")
            print(f"  Test:  mean={y_test_t.mean():.3f} std={y_test_t.std():.3f}")
            baseline = mae([y_train.mean().item()] * len(y_test_raw), y_test_raw)
            ds_results["baseline_mean_mae"] = baseline
            print(f"  Baseline (predict mean): MAE={baseline:.4f}")

        # Helper to record a probe result
        def record(name, preds_tensor):
            if task_type == "regression":
                m = mae(preds_tensor.tolist(), y_test_raw)
                ds_results[name] = {"mae": m}
                print(f"    {name:<28s} MAE={m:.4f}")
            else:
                pred_labels = [all_labels[p.item()] for p in preds_tensor]
                bal = balanced_accuracy(pred_labels, y_test_raw)
                ds_results[name] = {"balanced_accuracy": bal}
                print(f"    {name:<28s} bal_acc={bal:.3f}")

        # ─── Per-layer probes ───
        for layer in LAYERS:
            ln = f"L{layer}"
            print(f"\n  --- {ln} ---")

            # Mean-pool linear
            X_tr = mean_pool_vec(train_items, [layer])
            X_te = mean_pool_vec(test_items, [layer])
            preds = train_linear_probe(X_tr, y_train, X_te, n_classes,
                                       device=args.device)
            record(f"mean_linear_{ln}", preds)

            # Last-pos linear
            X_tr = last_pool_vec(train_items, [layer])
            X_te = last_pool_vec(test_items, [layer])
            preds = train_linear_probe(X_tr, y_train, X_te, n_classes,
                                       device=args.device)
            record(f"last_linear_{ln}", preds)

            # Attention probe (full sequence)
            preds = train_attn_probe(
                train_items, y_train, test_items, [layer],
                n_classes, n_heads=args.n_heads,
                lr=args.attn_lr, epochs=args.attn_epochs,
                batch_size=args.attn_batch_size, device=args.device,
            )
            record(f"attn_{ln}", preds)
            torch.cuda.empty_cache()

        # ─── Concat all layers ───
        print(f"\n  --- Concat (all layers) ---")

        # Mean-pool concat linear
        X_tr = mean_pool_vec(train_items, LAYERS)
        X_te = mean_pool_vec(test_items, LAYERS)
        preds = train_linear_probe(X_tr, y_train, X_te, n_classes,
                                   device=args.device)
        record("mean_linear_concat", preds)

        # Last-pos concat linear
        X_tr = last_pool_vec(train_items, LAYERS)
        X_te = last_pool_vec(test_items, LAYERS)
        preds = train_linear_probe(X_tr, y_train, X_te, n_classes,
                                   device=args.device)
        record("last_linear_concat", preds)

        # Attention probe concat
        preds = train_attn_probe(
            train_items, y_train, test_items, LAYERS,
            n_classes, n_heads=args.n_heads,
            lr=args.attn_lr, epochs=args.attn_epochs,
            batch_size=args.attn_batch_size, device=args.device,
        )
        record("attn_concat", preds)
        torch.cuda.empty_cache()

        all_results[ds_name] = ds_results

    # ── Summary Table ──
    print(f"\n\n{'=' * 110}")
    print("SUMMARY TABLE")
    print(f"{'=' * 110}")

    binary_ds = [d for d in datasets_to_run
                 if d in all_results and DATASETS[d][1] != "regression"]
    if binary_ds:
        print(f"\nClassification (balanced accuracy):")
        cols = ["mean_linear_L9", "mean_linear_L18", "mean_linear_L27",
                "mean_linear_concat",
                "last_linear_L9", "last_linear_L18", "last_linear_L27",
                "attn_L9", "attn_L18", "attn_L27", "attn_concat"]
        short = ["mL9", "mL18", "mL27", "mAll",
                 "lL9", "lL18", "lL27",
                 "aL9", "aL18", "aL27", "aAll"]
        header = f"{'Dataset':<22s} " + " ".join(f"{s:>5s}" for s in short) + "  N"
        print(header)
        print("-" * len(header))
        for ds_name in binary_ds:
            r = all_results[ds_name]
            vals = []
            for c in cols:
                v = r.get(c, {}).get("balanced_accuracy", 0)
                vals.append(f"{v:.3f}"[1:])  # strip leading 0
            print(f"{ds_name:<22s} " + " ".join(f"{v:>5s}" for v in vals)
                  + f"  {r['n_train']}")

    reg_ds = [d for d in datasets_to_run
              if d in all_results and DATASETS[d][1] == "regression"]
    if reg_ds:
        print(f"\nRegression (MAE, lower is better):")
        for ds_name in reg_ds:
            r = all_results[ds_name]
            print(f"  {ds_name}: baseline={r.get('baseline_mean_mae', 0):.4f}")
            for key in sorted(r.keys()):
                if isinstance(r[key], dict) and "mae" in r[key]:
                    print(f"    {key}: {r[key]['mae']:.4f}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()

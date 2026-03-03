#!/usr/bin/env python3
"""Train joint position-layer Qwen attention probe on 3 binary tasks.

Two modes:
  --extract-only   Extract and cache activations for all 3 tasks, then exit.
  --task TASK       Train probe on TASK, evaluate on all 3 tasks (cross-task transfer).

Tasks: hint_admission, sycophancy, truthfulqa_hint

Usage:
    # First run: extract activations for all tasks
    python scripts/train_qwen_probe.py --extract-only

    # Then train on each task (skip extraction since cached)
    python scripts/train_qwen_probe.py --task hint_admission --skip-extraction
    python scripts/train_qwen_probe.py --task sycophancy --skip-extraction
    python scripts/train_qwen_probe.py --task truthfulqa_hint --skip-extraction
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm, trange

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

load_dotenv(Path.home() / ".env")

FAST_CACHE_DIR = Path(os.environ["FAST_CACHE_DIR"])
CACHE_DIR = Path(os.environ["CACHE_DIR"])
ACT_CACHE_SUBSAMPLED = FAST_CACHE_DIR / "qwen_probe_acts_s1"
ACT_CACHE_FULL = FAST_CACHE_DIR / "qwen_probe_acts_s1_full"
ACT_CACHE = ACT_CACHE_FULL  # default to full (no subsampling)
CKPT_DIR = CACHE_DIR / "checkpoints" / "qwen_attention_probe"
LOG_DIR = _ROOT / "logs" / "qwen_probe"

LAYERS = [9, 18, 27]
STRIDE = 1
MAX_EXTRACT_POSITIONS = None  # None = no cap (keep all positions)
MODEL_NAME = "Qwen/Qwen3-8B"
D_MODEL = 4096
SEED = 42

# ── Task definitions ──

TASKS = {
    "hint_admission": {
        "hf_repo_train": "mats-10-sprint-cs-jb/qwen3-8b-hint-admission-rollouts",
        "hf_repo_test": "mats-10-sprint-cs-jb/cot-oracle-hint-admission-cleaned",
        "label_map": {"hint_used_correct": "influenced", "hint_used_wrong": "influenced", "hint_resisted": "independent"},
        "prompt_field": "hinted_prompt",
    },
    "sycophancy": {
        "hf_repo": "mats-10-sprint-cs-jb/cot-oracle-sycophancy-cleaned",
        "label_map": {"sycophantic": "influenced", "non_sycophantic": "independent"},
        "prompt_field": "hinted_prompt",
    },
    "truthfulqa_hint": {
        "hf_repo": "mats-10-sprint-cs-jb/cot-oracle-truthfulqa-hint-unverbalized-cleaned",
        "label_map": {"hint_used_correct": "influenced", "hint_used_wrong": "influenced", "hint_resisted": "independent"},
        "prompt_field": "hinted_prompt",
    },
}

BINARY_LABELS = ["independent", "influenced"]  # 0, 1
L2I = {l: i for i, l in enumerate(BINARY_LABELS)}

HF_ORG = "japhba"
WANDB_PROJECT = "cot_oracle"
WANDB_ENTITY = "MATS10-CS-JB"


# ── Model + probe imports ──

sys.path.insert(0, str(_ROOT / "baselines"))
from qwen_attention_probe import QwenAttentionProbe, _subsample_positions


class LinearConcatProbe(nn.Module):
    """Linear probe: pool positions per layer, concat layers, linear classify."""

    def __init__(self, layers: list[int], d_model: int = 4096, n_outputs: int = 2, pooling: str = "last"):
        super().__init__()
        self.layers = layers
        self.pooling = pooling
        self.linear = nn.Linear(len(layers) * d_model, n_outputs)

    def forward(self, inputs: list[dict[int, torch.Tensor]]) -> torch.Tensor:
        features = []
        for inp in inputs:
            if self.pooling == "last":
                layer_vecs = [inp[l][-1].to(dtype=torch.float32) for l in self.layers]
            else:
                layer_vecs = [inp[l].to(dtype=torch.float32).mean(dim=0) for l in self.layers]
            features.append(torch.cat(layer_vecs, dim=-1))
        return self.linear(torch.stack(features).to(self.linear.weight.device))


# ── Data loading ──

def _process_split(split, cfg: dict, task_name: str) -> list[dict]:
    """Convert a HF dataset split into list of item dicts with binarized labels."""
    items = []
    for row in split:
        raw_label = row["label"]
        binary = cfg["label_map"].get(raw_label)
        if binary is None:
            continue
        items.append({
            "prompt": row[cfg["prompt_field"]],
            "cot_text": row["cot_text"],
            "label": binary,
            "example_id": row.get("question_id", f"{task_name}_{len(items):05d}"),
        })
    return items


def load_task_data(task_name: str) -> tuple[list[dict], list[dict]]:
    """Load HF dataset, binarize labels. Returns (train_items, test_items)."""
    cfg = TASKS[task_name]

    if "hf_repo" in cfg:
        ds = load_dataset(cfg["hf_repo"])
        train_items = _process_split(ds["train"], cfg, task_name)
        test_items = _process_split(ds["test"], cfg, task_name)
    else:
        train_split = load_dataset(cfg["hf_repo_train"], split="train")
        test_split = load_dataset(cfg["hf_repo_test"], split="test")
        train_items = _process_split(train_split, cfg, task_name)
        test_items = _process_split(test_split, cfg, task_name)

    dist_tr = Counter(it["label"] for it in train_items)
    dist_te = Counter(it["label"] for it in test_items)
    print(f"  {task_name}: train={len(train_items)} {dict(dist_tr)}, test={len(test_items)} {dict(dist_te)}")
    return train_items, test_items


# ── Activation extraction + caching ──

def build_generation_prompt(prompt: str, tokenizer) -> str:
    """Reconstruct chat-template prompt matching generation time."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def _subsample_indices(K: int, max_k: int) -> list[int]:
    """Return indices for uniform subsampling of K positions to max_k, always keeping last."""
    if K <= max_k:
        return list(range(K))
    idx = torch.linspace(0, K - 2, max_k - 1).long().tolist()
    idx.append(K - 1)
    return idx


def extract_and_cache_activations(
    items: list[dict], task_name: str, split_name: str,
    model, tokenizer, device: str = "cuda",
) -> list[dict]:
    """Extract per-position multi-layer activations, cache to disk as .pt files.

    Uses stride=1 (every CoT token), subsampled to MAX_EXTRACT_POSITIONS per layer.
    Also stores token_ids at each position for later decoding.
    """
    from core.ao import EarlyStopException, get_hf_submodule
    from cot_utils import get_cot_stride_positions

    cache_dir = ACT_CACHE / task_name / split_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    successful = []
    for item in tqdm(items, desc=f"Extracting {task_name}/{split_name}"):
        cache_path = cache_dir / f"{item['example_id']}.pt"

        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            item["acts"] = {k: v for k, v in cached.items() if isinstance(k, int)}
            item["token_ids"] = cached["token_ids"]
            successful.append(item)
            continue

        formatted = build_generation_prompt(item["prompt"], tokenizer)
        full_text = formatted + item["cot_text"]

        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        all_ids = tokenizer.encode(full_text, add_special_tokens=False)
        positions = get_cot_stride_positions(len(prompt_ids), len(all_ids), stride=STRIDE)

        if len(positions) < 2:
            continue

        tok_out = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        input_tensor = tok_out["input_ids"].to(device)
        attn_mask = tok_out["attention_mask"].to(device)
        seq_len = input_tensor.shape[1]
        positions = [p for p in positions if p < seq_len]
        if len(positions) < 2:
            continue

        # Subsample positions if cap is set
        if MAX_EXTRACT_POSITIONS and len(positions) > MAX_EXTRACT_POSITIONS:
            sub_idx = _subsample_indices(len(positions), MAX_EXTRACT_POSITIONS)
            positions = [positions[i] for i in sub_idx]

        # Get token IDs at each position
        token_ids_at_pos = input_tensor[0, positions].cpu()

        # Extract activations via hooks
        submodules = {l: get_hf_submodule(model, l) for l in LAYERS}
        max_layer = max(LAYERS)
        acts = {}
        mod_to_layer = {id(s): l for l, s in submodules.items()}

        def hook_fn(module, _inputs, outputs):
            layer = mod_to_layer[id(module)]
            raw = outputs[0] if isinstance(outputs, tuple) else outputs
            acts[layer] = raw[0, positions, :].detach().cpu()
            if layer == max_layer:
                raise EarlyStopException()

        handles = [s.register_forward_hook(hook_fn) for s in submodules.values()]
        with torch.no_grad():
            try:
                model(input_ids=input_tensor, attention_mask=attn_mask)
            except EarlyStopException:
                pass
            finally:
                for h in handles:
                    h.remove()

        # Save acts + token_ids together
        save_dict = {**acts, "token_ids": token_ids_at_pos}
        torch.save(save_dict, cache_path)
        item["acts"] = acts
        item["token_ids"] = token_ids_at_pos
        successful.append(item)

    print(f"  {task_name}/{split_name}: {len(successful)}/{len(items)} extracted")
    return successful


def load_cached_activations(items: list[dict], task_name: str, split_name: str) -> list[dict]:
    """Load cached activations from disk."""
    cache_dir = ACT_CACHE / task_name / split_name
    successful = []
    for item in items:
        cache_path = cache_dir / f"{item['example_id']}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            item["acts"] = {k: v for k, v in cached.items() if isinstance(k, int)}
            item["token_ids"] = cached["token_ids"]
            successful.append(item)
    print(f"  {task_name}/{split_name}: {len(successful)}/{len(items)} loaded from cache")
    return successful


# ── Training ──

def train_probe(
    model: nn.Module, train_items: list[dict], val_items: list[dict], *,
    lr: float = 1e-4, epochs: int = 50, patience: int = 20,
    batch_size: int = 32, device: str = "cuda", wandb_run=None,
) -> nn.Module:
    """Train probe with val-based early stopping. Returns best model."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_acts = [it["acts"] for it in train_items]
    train_y = torch.tensor([L2I[it["label"]] for it in train_items], device=device)
    val_acts = [it["acts"] for it in val_items]
    val_y = torch.tensor([L2I[it["label"]] for it in val_items], device=device)

    best_val_loss, patience_ctr, best_state = float("inf"), 0, None

    for ep in trange(epochs, desc="Training"):
        model.train()
        perm = torch.randperm(len(train_acts))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_acts), batch_size):
            idx = perm[start:start + batch_size]
            batch_acts = [train_acts[i] for i in idx]
            batch_y = train_y[idx]

            opt.zero_grad(set_to_none=True)
            logits = model(batch_acts).float()
            loss = loss_fn(logits, batch_y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Val loss
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for start in range(0, len(val_acts), batch_size):
                batch = val_acts[start:start + batch_size]
                batch_y_v = val_y[start:start + batch_size]
                logits = model(batch).float()
                val_loss += loss_fn(logits, batch_y_v).item()
                n_val_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        avg_val = val_loss / max(n_val_batches, 1)

        if wandb_run:
            wandb_run.log({"train/loss": avg_train, "val/loss": avg_val, "epoch": ep})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {ep} (best val_loss={best_val_loss:.4f})")
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def evaluate_probe(
    model: nn.Module, items: list[dict], task_name: str,
    batch_size: int = 32, device: str = "cuda",
) -> dict:
    """Evaluate probe on a test set. Returns metrics dict + per-example predictions."""
    model.eval()
    acts = [it["acts"] for it in items]
    y_true = [L2I[it["label"]] for it in items]

    all_preds = []
    with torch.no_grad():
        for start in range(0, len(acts), batch_size):
            batch = acts[start:start + batch_size]
            logits = model(batch)
            all_preds.append(logits.argmax(1).cpu())
    preds = torch.cat(all_preds).tolist()

    pred_labels = [BINARY_LABELS[p] for p in preds]
    true_labels = [BINARY_LABELS[y] for y in y_true]

    acc = accuracy_score(true_labels, pred_labels)
    bal_acc = balanced_accuracy_score(true_labels, pred_labels)
    prec, rec, f1, sup = precision_recall_fscore_support(
        true_labels, pred_labels, labels=BINARY_LABELS, zero_division=0)

    metrics = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "n_items": len(items),
        "per_class": {
            label: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                     "f1": round(float(f1[i]), 4), "support": int(sup[i])}
            for i, label in enumerate(BINARY_LABELS)
        },
    }

    examples = []
    for i, item in enumerate(items):
        examples.append({
            "example_id": item["example_id"],
            "true_label": true_labels[i],
            "pred_label": pred_labels[i],
            "correct": true_labels[i] == pred_labels[i],
        })

    return metrics, examples


# ── Max-activating token analysis ──

def log_max_activating_tokens(
    model: QwenAttentionProbe, items: list[dict], tokenizer,
    wandb_run, task_name: str, n_examples: int = 50, top_k: int = 10, device: str = "cuda",
):
    """Log top-K most-attended tokens per example to wandb as a Table."""
    import wandb

    model.eval()
    rows = []

    with torch.no_grad():
        for item in tqdm(items[:n_examples], desc=f"Attention analysis ({task_name})"):
            acts = item["acts"]
            tok_ids = item["token_ids"]  # [K]

            # Get the subsampled K that the probe uses
            K = tok_ids.shape[0]
            max_k = model.max_positions_per_layer
            sub_idx = _subsample_indices(K, max_k)
            tok_ids_sub = tok_ids[sub_idx]
            K_sub = len(sub_idx)

            # Forward with attention weights (batch of 1)
            logits, attn_w, valid = model([acts], return_attention=True)
            # attn_w: [1, T, T] where T = n_layers * K_sub_padded
            # valid: [1, T]

            attn_w = attn_w[0]  # [T, T]
            valid_mask = valid[0].float()  # [T]

            # Key importance: how much each position is attended TO (column sum, masked)
            importance = (attn_w * valid_mask.unsqueeze(0)).sum(dim=0)  # [T]

            # Group by CoT position (average across layers)
            # Joint seq is [layer0 K positions, layer1 K positions, layer2 K positions]
            n_layers = len(model.layers)
            total_valid = int(valid_mask.sum().item())
            # Each layer contributes K_sub positions (they share the same positions)
            if total_valid >= n_layers * K_sub:
                layer_imp = importance[:n_layers * K_sub].view(n_layers, K_sub)
                pos_importance = layer_imp.mean(dim=0)  # [K_sub]
            else:
                pos_importance = importance[:K_sub]

            # Top-K positions
            actual_top_k = min(top_k, K_sub)
            top_vals, top_idx = pos_importance.topk(actual_top_k)
            top_token_ids = tok_ids_sub[top_idx.cpu()].tolist()
            top_tokens = tokenizer.convert_ids_to_tokens(top_token_ids)

            pred = logits.argmax(1).item()
            true = L2I[item["label"]]

            tokens_str = ", ".join(f"{t} ({v:.3f})" for t, v in zip(top_tokens, top_vals.tolist()))
            rows.append([
                item["example_id"], BINARY_LABELS[true], BINARY_LABELS[pred],
                pred == true, tokens_str,
            ])

    table = wandb.Table(
        columns=["example_id", "true_label", "pred_label", "correct", "top_tokens"],
        data=rows,
    )
    wandb_run.log({f"max_activating_tokens/{task_name}": table})
    print(f"  Logged {len(rows)} examples with max-activating tokens for {task_name}")


def save_checkpoint_and_upload(model: nn.Module, task_name: str, probe_type: str = "attention"):
    """Save checkpoint to /ceph and upload to HuggingFace."""
    ckpt_dir = CKPT_DIR / f"{probe_type}_{task_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    # Upload to HF
    from huggingface_hub import HfApi
    api = HfApi()
    repo_id = f"{HF_ORG}/qwen-{probe_type}-probe-{task_name}"
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_file(path_or_fileobj=str(ckpt_path), path_in_repo="model.pt", repo_id=repo_id)
    print(f"  Uploaded to https://huggingface.co/{repo_id}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Train joint Qwen attention probe")
    parser.add_argument("--task", type=str, choices=list(TASKS.keys()),
                        help="Task to train on (evaluates all 3 tasks)")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract + cache activations, don't train")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Use cached activations only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--probe-type", type=str, choices=["attention", "linear"], default="attention",
                        help="Probe type: 'attention' (joint QwenAttentionProbe) or 'linear' (concat linear)")
    parser.add_argument("--pooling", type=str, choices=["last", "mean"], default="last",
                        help="Pooling strategy for linear probe (default: last)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if not args.extract_only and not args.task:
        parser.error("Must specify --task or --extract-only")

    torch.manual_seed(args.seed)

    # ── Load datasets ──
    print("Loading datasets...")
    all_data = {}
    for task_name in TASKS:
        all_data[task_name] = load_task_data(task_name)

    # ── Activation extraction ──
    if not args.skip_extraction:
        print("\nLoading Qwen3-8B for activation extraction...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from core.ao import choose_attn_implementation

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map=args.device,
            attn_implementation=choose_attn_implementation(MODEL_NAME))
        model.eval()

        for task_name, (train_items, test_items) in all_data.items():
            train_items = extract_and_cache_activations(
                train_items, task_name, "train", model, tokenizer, device=args.device)
            test_items = extract_and_cache_activations(
                test_items, task_name, "test", model, tokenizer, device=args.device)
            all_data[task_name] = (train_items, test_items)

        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print("  LLM freed.\n")
    else:
        print("\nLoading cached activations...")
        for task_name, (train_items, test_items) in all_data.items():
            train_items = load_cached_activations(train_items, task_name, "train")
            test_items = load_cached_activations(test_items, task_name, "test")
            all_data[task_name] = (train_items, test_items)

    if args.extract_only:
        print("Extraction complete.")
        return

    # ── Train on specified task ──
    task = args.task
    train_items, test_items = all_data[task]
    print(f"\n{'='*60}")
    print(f"  Training probe on: {task}")
    print(f"  Train: {len(train_items)}, Test: {len(test_items)}")
    print(f"{'='*60}")

    # 85/15 train/val split
    n_val = max(1, int(len(train_items) * args.val_frac))
    perm = torch.randperm(len(train_items), generator=torch.Generator().manual_seed(args.seed))
    val_items = [train_items[i] for i in perm[:n_val]]
    actual_train = [train_items[i] for i in perm[n_val:]]
    print(f"  Split: {len(actual_train)} train, {len(val_items)} val")

    # Create model
    if args.probe_type == "linear":
        probe_model = LinearConcatProbe(LAYERS, n_outputs=2, pooling=args.pooling).to(device=args.device)
        run_name = f"linear-{args.pooling}-concat-{task}"
        wandb_group = "probes"
        default_lr = args.lr if args.lr != 1e-4 else 0.01  # higher default for linear
    else:
        probe_model = QwenAttentionProbe(LAYERS, n_outputs=2).to(device=args.device, dtype=torch.bfloat16)
        run_name = f"qwen-probe-{task}"
        wandb_group = "attprobes"
        default_lr = args.lr

    # wandb
    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            group=wandb_group,
            name=run_name,
            config={"task": task, "probe_type": args.probe_type, "lr": default_lr,
                    "epochs": args.epochs, "patience": args.patience,
                    "batch_size": args.batch_size, "layers": LAYERS, "stride": STRIDE,
                    "val_frac": args.val_frac, "max_extract_positions": MAX_EXTRACT_POSITIONS},
        )

    # Train
    probe = train_probe(
        probe_model, actual_train, val_items,
        lr=default_lr, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, device=args.device, wandb_run=wandb_run,
    )

    # ── Evaluate on all 3 tasks ──
    print(f"\n{'='*60}")
    print("  Cross-task evaluation (3x3 transfer matrix)")
    print(f"{'='*60}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    transfer_results = {}

    # Load tokenizer for max-activating token analysis
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for eval_task in TASKS:
        _, eval_test = all_data[eval_task]
        if not eval_test:
            print(f"  {eval_task}: no test data, skipping")
            continue

        metrics, examples = evaluate_probe(probe, eval_test, eval_task, device=args.device)
        transfer_results[eval_task] = metrics
        print(f"  {task} → {eval_task}: bal_acc={metrics['balanced_accuracy']:.4f}, "
              f"acc={metrics['accuracy']:.4f}, n={metrics['n_items']}")

        if wandb_run:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    wandb_run.log({f"eval/{eval_task}/{k}": v})
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, dict):
                            for k3, v3 in v2.items():
                                if isinstance(v3, (int, float)):
                                    wandb_run.log({f"eval/{eval_task}/{k2}/{k3}": v3})

            # Log max-activating tokens for attention probes only
            if args.probe_type == "attention":
                log_max_activating_tokens(probe, eval_test, tokenizer, wandb_run, eval_task, device=args.device)

        # Write per-example JSONL logfile
        log_path = LOG_DIR / f"{task}_eval_{eval_task}.jsonl"
        with open(log_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"    Logfile: {log_path}")

    # Save transfer summary
    summary_path = LOG_DIR / f"{task}_transfer_summary.json"
    with open(summary_path, "w") as f:
        json.dump(transfer_results, f, indent=2)
    print(f"\n  Transfer summary: {summary_path}")

    # Log transfer matrix to wandb
    if wandb_run:
        import wandb
        table = wandb.Table(columns=["eval_task", "balanced_accuracy", "accuracy", "n_items"])
        for eval_task, m in transfer_results.items():
            table.add_data(eval_task, m["balanced_accuracy"], m["accuracy"], m["n_items"])
        wandb_run.log({f"transfer_matrix/{task}": table})

    # ── Save + upload ──
    save_checkpoint_and_upload(probe, task, probe_type=args.probe_type)

    if wandb_run:
        wandb_run.finish()

    print(f"\nDone: {task}")


if __name__ == "__main__":
    main()

"""
Train CoT Oracle v2: Single-Layer, New Task Mix

Changes from v1:
  - Single layer only (50% depth) — multi-layer showed no improvement
  - Placeholder token: ¶ instead of ? (? collides with questions)
  - New tasks: answer_prediction, full_reconstruction, causal_prediction, conversational
  - Dropped: summary (broken labels), multi-layer machinery
  - Kept: context_prediction, decorative, domain, correctness, persona

Task mix (~225K examples):
  1. Context prediction — random positions (100K) — 1 random layer from [25/50/75%]
  2. Answer prediction — stepwise (20K) — predict final answer from prefix activations
  3. Full reconstruction (15K) — predict entire CoT from all boundary activations
  4. Causal prediction (30K) — autoregressive: feed 1..i, predict next chunk
  5. Conversational QA (5K) — 107 question types, Gemini Flash responses
  6. Decorative CoT (10K) — load-bearing vs decorative binary
  7. Domain classification (15K) — multi-class
  8. Correctness prediction (15K) — binary correct/incorrect
  9. Persona detection (15K) — multi-class unverbalized prompt

Usage:
    torchrun --nproc_per_node=1 src/train_v2.py \
        --corpus data/cot_corpus_v5/mini_corpus.jsonl \
        --conv-qa data/hf_upload/cot_conversational_qa.jsonl \
        --model Qwen/Qwen3-8B
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.ao_repo import ensure_ao_repo_on_path

ensure_ao_repo_on_path()

import torch

import nl_probes.utils.dataset_utils as du_module
from nl_probes.utils.dataset_utils import (
    create_training_datapoint,
    TrainingDataPoint,
    find_pattern_in_tokens,
)
import nl_probes.sft as sft_module
from nl_probes.sft import train_model
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.utils.common import load_tokenizer

# ── Override placeholder token ──
# " ?" collides with question marks in conversational QA prompts.
# " ¶" (pilcrow, token 78846) has zero occurrences in corpus + QA data.
PLACEHOLDER_TOKEN = " ¶"

# Monkey-patch AO's SPECIAL_TOKEN everywhere
du_module.SPECIAL_TOKEN = PLACEHOLDER_TOKEN

# Also patch get_introspection_prefix to use our token
_orig_get_prefix = du_module.get_introspection_prefix
def _patched_get_prefix(sae_layer: int, num_positions: int) -> str:
    prefix = f"Layer: {sae_layer}\n"
    prefix += PLACEHOLDER_TOKEN * num_positions
    prefix += " \n"
    return prefix
du_module.get_introspection_prefix = _patched_get_prefix

# Our dataset loaders
from dataset_classes.cot_context_prediction import load_cot_context_prediction_data
from dataset_classes.cot_answer_prediction import load_cot_answer_prediction_data
from dataset_classes.cot_full_reconstruction import load_cot_full_reconstruction_data
from dataset_classes.cot_causal_prediction import load_cot_causal_prediction_data
from dataset_classes.cot_conversational import load_cot_conversational_data
from dataset_classes.cot_decorative import load_cot_decorative_data
from dataset_classes.cot_domain import load_cot_domain_data
from dataset_classes.cot_correctness import load_cot_correctness_data
from dataset_classes.cot_persona import load_cot_persona_data

from cot_utils import (
    find_sentence_boundary_positions,
    layer_percent_to_layer,
    split_cot_into_sentences,
)


def ensure_boundary_positions(corpus_path: str, tokenizer) -> str:
    """Ensure all corpus entries have boundary_positions computed."""
    import json

    entries = []
    needs_update = False
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                entries.append(entry)
                if not entry.get("boundary_positions"):
                    needs_update = True

    if not needs_update:
        print(f"All {len(entries)} corpus entries already have boundary_positions")
        return corpus_path

    print(f"Computing boundary_positions for {len(entries)} entries...")
    updated = 0
    for entry in entries:
        if entry.get("boundary_positions"):
            continue

        sentences = entry.get("sentences") or split_cot_into_sentences(entry["cot_response"])
        if len(sentences) < 2:
            continue

        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        full_text = formatted + entry["cot_response"]
        boundary_positions = find_sentence_boundary_positions(tokenizer, full_text, sentences)

        entry["boundary_positions"] = boundary_positions
        entry["sentences"] = sentences
        entry["n_sentences"] = len(sentences)
        updated += 1

    with open(corpus_path, "w") as f:
        for entry in entries:
            import json as _json
            f.write(_json.dumps(entry) + "\n")

    print(f"  Updated {updated} entries with boundary_positions")
    return corpus_path


def dicts_to_training_data(
    raw_data: list[dict],
    tokenizer,
) -> list[TrainingDataPoint]:
    """Convert dataset loader output to AO TrainingDataPoint objects.

    v2: Single-layer only. All items have 'layer' (int).
    No multi-layer doubling.
    """
    training_data = []
    skipped = 0

    for item in raw_data:
        try:
            # v2: all items are single-layer
            layer = item.get("layer")
            if layer is None:
                # Backwards compat: if 'layers' list, take middle
                layers = item.get("layers", [])
                if layers:
                    layer = layers[len(layers) // 2]
                else:
                    skipped += 1
                    continue

            dp = create_training_datapoint(
                datapoint_type=item["datapoint_type"],
                prompt=item["prompt"],
                target_response=item["target_response"],
                layer=layer,
                num_positions=item["num_positions"],
                tokenizer=tokenizer,
                acts_BD=None,
                feature_idx=-1,
                context_input_ids=item["context_input_ids"],
                context_positions=item["context_positions"],
            )
            training_data.append(dp)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Warning: skipped datapoint ({e})")

    if skipped > 0:
        print(f"  Skipped {skipped} datapoints during conversion")

    return training_data


def build_training_mixture(
    corpus_path: str,
    conv_qa_path: str | None,
    persona_corpus_path: str | None,
    tokenizer,
    model_name: str,
    layer_percents: list[int],
    task_sizes: dict[str, int],
) -> list[TrainingDataPoint]:
    """Build the v2 mixed training data."""

    all_data = []

    # Task 1: Context Prediction — Random Positions (backbone)
    print("\n=== Task 1: Context Prediction — Random Positions ===")
    try:
        raw = load_cot_context_prediction_data(
            corpus_path, tokenizer, model_name, layer_percents,
            num_examples=task_sizes.get("context_prediction", 100000),
        )
        data = dicts_to_training_data(raw, tokenizer)
        print(f"  -> {len(data)} examples")
        all_data.extend(data)
    except Exception as e:
        print(f"  FAILED: {e}")

    # Task 2: Answer Prediction — Stepwise
    print("\n=== Task 2: Answer Prediction — Stepwise ===")
    try:
        raw = load_cot_answer_prediction_data(
            corpus_path, tokenizer, model_name, layer_percents,
            num_examples=task_sizes.get("answer_prediction", 20000),
        )
        data = dicts_to_training_data(raw, tokenizer)
        print(f"  -> {len(data)} examples")
        all_data.extend(data)
    except Exception as e:
        print(f"  FAILED: {e}")

    # Task 3: Full CoT Reconstruction
    print("\n=== Task 3: Full CoT Reconstruction ===")
    try:
        raw = load_cot_full_reconstruction_data(
            corpus_path, tokenizer, model_name, layer_percents,
            num_examples=task_sizes.get("full_reconstruction", 15000),
        )
        data = dicts_to_training_data(raw, tokenizer)
        print(f"  -> {len(data)} examples")
        all_data.extend(data)
    except Exception as e:
        print(f"  FAILED: {e}")

    # Task 4: Causal Prediction — Stage-wise
    print("\n=== Task 4: Causal Prediction — Stage-wise ===")
    try:
        raw = load_cot_causal_prediction_data(
            corpus_path, tokenizer, model_name, layer_percents,
            num_examples=task_sizes.get("causal_prediction", 30000),
        )
        data = dicts_to_training_data(raw, tokenizer)
        print(f"  -> {len(data)} examples")
        all_data.extend(data)
    except Exception as e:
        print(f"  FAILED: {e}")

    # Task 5: Conversational QA
    if conv_qa_path and Path(conv_qa_path).exists():
        print("\n=== Task 5: Conversational QA ===")
        try:
            raw = load_cot_conversational_data(
                corpus_path, conv_qa_path, tokenizer, model_name, layer_percents,
                num_examples=task_sizes.get("conversational", 5000),
            )
            data = dicts_to_training_data(raw, tokenizer)
            print(f"  -> {len(data)} examples")
            all_data.extend(data)
        except Exception as e:
            print(f"  FAILED: {e}")
    else:
        print(f"\n  Skipping Task 5 (no conversational QA at {conv_qa_path})")

    # Task 6: Decorative CoT
    print("\n=== Task 6: Decorative CoT ===")
    try:
        raw = load_cot_decorative_data(
            corpus_path, tokenizer, model_name, layer_percents,
            num_examples=task_sizes.get("decorative", 10000),
        )
        data = dicts_to_training_data(raw, tokenizer)
        print(f"  -> {len(data)} examples")
        all_data.extend(data)
    except ValueError as e:
        print(f"  Skipped ({e})")

    # Task 7: Domain Classification
    print("\n=== Task 7: Domain Classification ===")
    try:
        raw = load_cot_domain_data(
            corpus_path, tokenizer, model_name, layer_percents,
            num_examples=task_sizes.get("domain", 15000),
        )
        data = dicts_to_training_data(raw, tokenizer)
        print(f"  -> {len(data)} examples")
        all_data.extend(data)
    except Exception as e:
        print(f"  FAILED: {e}")

    # Task 8: Correctness Prediction
    print("\n=== Task 8: Correctness Prediction ===")
    try:
        raw = load_cot_correctness_data(
            corpus_path, tokenizer, model_name, layer_percents,
            num_examples=task_sizes.get("correctness", 15000),
        )
        data = dicts_to_training_data(raw, tokenizer)
        print(f"  -> {len(data)} examples")
        all_data.extend(data)
    except Exception as e:
        print(f"  FAILED: {e}")

    # Task 9: Persona Detection
    if persona_corpus_path and Path(persona_corpus_path).exists():
        print("\n=== Task 9: Persona Detection ===")
        try:
            raw = load_cot_persona_data(
                persona_corpus_path, tokenizer, model_name, layer_percents,
                num_examples=task_sizes.get("persona", 15000),
            )
            data = dicts_to_training_data(raw, tokenizer)
            print(f"  -> {len(data)} examples")
            all_data.extend(data)
        except Exception as e:
            print(f"  FAILED: {e}")
    else:
        print(f"\n  Skipping Task 9 (no persona corpus at {persona_corpus_path})")

    print(f"\n{'=' * 60}")
    print(f"Total training examples: {len(all_data)}")

    type_counts = Counter(dp.datapoint_type for dp in all_data)
    for dtype, count in sorted(type_counts.items()):
        pct = count / len(all_data) * 100
        print(f"  {dtype}: {count} ({pct:.1f}%)")

    return all_data


def install_fuzzy_eval_hook():
    """Monkey-patch AO's eval to use task-aware scoring instead of exact match.

    Classification tasks: substring match (is the label in the response?)
    Generation tasks: token F1 (overlap between predicted and target tokens)
    Also logs sample responses for qualitative inspection.
    """
    import wandb
    from nl_probes.utils.eval import run_evaluation, parse_answer

    # Tasks where exact match makes sense (short labels)
    CLASSIFICATION_TASKS = {
        "cot_decorative", "cot_domain", "cot_correctness", "cot_persona",
    }
    # Tasks where we need more generation tokens and fuzzy scoring
    GENERATION_TASKS = {
        "cot_full_reconstruction", "cot_causal_prediction",
        "cot_conversational", "cot_context_prediction",
    }
    # Answer prediction: short numeric answers, exact match is OK but strip whitespace
    ANSWER_TASKS = {"cot_answer_prediction"}

    def token_f1(prediction: str, reference: str) -> float:
        """Compute token-level F1 between prediction and reference."""
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        if not pred_tokens or not ref_tokens:
            return 0.0
        common = pred_tokens & ref_tokens
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def fuzzy_score(ds_name, responses, eval_data):
        """Score eval responses with task-aware metrics."""
        scores = []
        samples = []

        for resp, dp in zip(responses, eval_data):
            pred = resp.api_response.strip()
            target = dp.target_output.strip()

            if ds_name in CLASSIFICATION_TASKS:
                # Substring match: is the target label in the response?
                pred_lower = pred.lower()
                target_lower = target.lower()
                score = 1.0 if target_lower in pred_lower else 0.0
            elif ds_name in ANSWER_TASKS:
                # Normalized exact match for short answers
                pred_clean = parse_answer(pred)
                target_clean = parse_answer(target)
                score = 1.0 if pred_clean == target_clean else 0.0
            else:
                # Token F1 for generation tasks
                score = token_f1(pred, target)

            scores.append(score)
            if len(samples) < 3:
                samples.append((pred[:200], target[:200], score))

        avg = sum(scores) / len(scores) if scores else 0.0
        return avg, samples

    _original_eval = sft_module.eval_all_datasets

    def patched_eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step):
        """Task-aware eval with fuzzy scoring (replaces AO's yes/no format check)."""
        model.eval()
        eval_results = {}

        gen_kwargs_short = {"do_sample": False, "max_new_tokens": 20}
        gen_kwargs_long = {"do_sample": False, "max_new_tokens": 100}

        for ds in eval_datasets:
            gen_kwargs = gen_kwargs_long if ds in GENERATION_TASKS else gen_kwargs_short

            eval_responses = run_evaluation(
                eval_data=eval_datasets[ds],
                model=model,
                tokenizer=tokenizer,
                submodule=submodule,
                device=device,
                dtype=dtype,
                global_step=global_step,
                lora_path=None,
                eval_batch_size=cfg.eval_batch_size,
                steering_coefficient=cfg.steering_coefficient,
                generation_kwargs=gen_kwargs,
            )

            fuzzy_acc, samples = fuzzy_score(ds, eval_responses, eval_datasets[ds])
            task_type = "cls" if ds in CLASSIFICATION_TASKS else ("ans" if ds in ANSWER_TASKS else "gen")
            eval_results[f"eval/{ds}"] = fuzzy_acc
            eval_results[f"eval_{task_type}/{ds}"] = fuzzy_acc
            print(f"  Step {global_step} | {ds} ({task_type}): {fuzzy_acc:.1%}")

            if samples:
                pred, target, score = samples[0]
                print(f"    pred='{pred[:120]}' | target='{target[:120]}' | {score:.2f}")

        wandb.log(eval_results, step=global_step)
        wandb.summary.update(eval_results)
        model.train()
        torch.cuda.empty_cache()

    sft_module.eval_all_datasets = patched_eval_all_datasets
    print("Installed fuzzy eval scoring hook")


def install_per_task_loss_hook():
    """Monkey-patch AO's training loop to log per-task loss to wandb."""
    import wandb
    import torch.nn.functional as F
    from nl_probes.sft import train_features_batch as _original_train
    from nl_probes.utils.steering_hooks import get_hf_activation_steering_hook, add_hook

    _batch_state = {"types": []}

    from nl_probes.sft import construct_batch as _original_construct
    def patched_construct_batch(batch_list, tokenizer, device):
        _batch_state["types"] = [dp.datapoint_type for dp in batch_list]
        return _original_construct(batch_list, tokenizer, device)
    sft_module.construct_batch = patched_construct_batch

    def patched_train_features_batch(cfg, training_batch, model, submodule, device, dtype):
        hook_fn = get_hf_activation_steering_hook(
            vectors=training_batch.steering_vectors,
            positions=training_batch.positions,
            steering_coefficient=cfg.steering_coefficient,
            device=device,
            dtype=dtype,
        )
        tokenized_input = {
            "input_ids": training_batch.input_ids,
            "attention_mask": training_batch.attention_mask,
        }
        with add_hook(submodule, hook_fn):
            outputs = model(**tokenized_input, labels=training_batch.labels)

        batch_types = _batch_state["types"]
        if batch_types and len(batch_types) == training_batch.input_ids.shape[0]:
            logits = outputs.logits
            labels = training_batch.labels
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none',
            ).view(shift_labels.shape)
            mask = (shift_labels != -100).float()
            per_item_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            task_losses = defaultdict(list)
            for i, task_type in enumerate(batch_types):
                task_losses[task_type].append(per_item_loss[i].item())

            log_dict = {}
            for task, losses in task_losses.items():
                log_dict[f"train/loss_{task}"] = sum(losses) / len(losses)
            if wandb.run is not None:
                wandb.log(log_dict, commit=False)

        return outputs.loss

    sft_module.train_features_batch = patched_train_features_batch
    print("Installed per-task loss logging hook")


def install_unfaithfulness_eval_hook(model_name, eval_dir="data/evals", fast_n=5):
    """Monkey-patch AO's eval to run unfaithfulness evals alongside training evals."""
    import wandb
    from evals.common import load_eval_items
    from evals.score_oracle import score_eval, EVAL_PARSING
    from evals.run_evals import run_single_item

    eval_dir = Path(eval_dir)
    act_layer = layer_percent_to_layer(model_name, 50)

    fast_items = {}
    for eval_file in sorted(eval_dir.glob("*.json")):
        eval_name = eval_file.stem
        if eval_name in ("decorative_cot", "sentence_insertion"):
            continue
        items = load_eval_items(eval_file)
        fast_items[eval_name] = items[:fast_n]

    total_items = sum(len(v) for v in fast_items.values())
    print(f"Unfaithfulness eval hook: {len(fast_items)} evals, {total_items} total items")

    _original_eval = sft_module.eval_all_datasets

    def patched_eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step):
        _original_eval(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step)

        print(f"\n--- Unfaithfulness Evals (step {global_step}) ---")

        for eval_name, items in fast_items.items():
            try:
                completed = []
                for item in items:
                    result = run_single_item(
                        model, tokenizer, item, act_layer,
                        model_name=model_name, device=str(device),
                    )
                    completed.append(result)

                parsing_config = EVAL_PARSING.get(eval_name)
                if parsing_config:
                    metrics = score_eval(eval_name, completed, parsing_config)
                    if metrics:
                        wandb.log({
                            f"unfaith/{eval_name}/accuracy": metrics["accuracy"],
                            f"unfaith/{eval_name}/n_scored": metrics.get("n_items", 0),
                        }, step=global_step)
                        print(f"  {eval_name}: acc={metrics['accuracy']:.3f} ({metrics.get('n_items', 0)} scored)")

            except Exception as e:
                print(f"  {eval_name}: FAILED ({e})")

        print("--- End Unfaithfulness Evals ---\n")

    sft_module.eval_all_datasets = patched_eval_all_datasets
    print("Installed unfaithfulness eval hook")


def main():
    parser = argparse.ArgumentParser(description="Train CoT Oracle v2")
    parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--conv-qa", default=None, help="Path to conversational QA JSONL")
    parser.add_argument("--persona-corpus", default=None, help="Path to persona corpus.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-dir", default="checkpoints/cot_oracle_v2")
    parser.add_argument("--wandb-project", default="cot_oracle")
    parser.add_argument("--wandb-run", default="")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.add_argument("--eval-dir", default="data/evals")
    parser.add_argument("--fast-eval-n", type=int, default=10)
    parser.add_argument("--no-unfaith-evals", action="store_true")

    # Task size overrides
    parser.add_argument("--n-context-pred", type=int, default=100000)
    parser.add_argument("--n-answer-pred", type=int, default=20000)
    parser.add_argument("--n-full-recon", type=int, default=15000)
    parser.add_argument("--n-causal-pred", type=int, default=30000)
    parser.add_argument("--n-conversational", type=int, default=5000)
    parser.add_argument("--n-decorative", type=int, default=10000)
    parser.add_argument("--n-domain", type=int, default=15000)
    parser.add_argument("--n-correctness", type=int, default=15000)
    parser.add_argument("--n-persona", type=int, default=15000)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    layer_percents = [25, 50, 75]

    # Verify placeholder token
    tok_ids = tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)
    assert len(tok_ids) == 1, f"Placeholder '{PLACEHOLDER_TOKEN}' is {len(tok_ids)} tokens, need 1"
    print(f"Placeholder token: '{PLACEHOLDER_TOKEN}' -> token ID {tok_ids[0]}")

    task_sizes = {
        "context_prediction": args.n_context_pred,
        "answer_prediction": args.n_answer_pred,
        "full_reconstruction": args.n_full_recon,
        "causal_prediction": args.n_causal_pred,
        "conversational": args.n_conversational,
        "decorative": args.n_decorative,
        "domain": args.n_domain,
        "correctness": args.n_correctness,
        "persona": args.n_persona,
    }

    # Ensure boundary_positions
    print("Ensuring boundary_positions are computed...")
    ensure_boundary_positions(args.corpus, tokenizer)
    if args.persona_corpus and Path(args.persona_corpus).exists():
        ensure_boundary_positions(args.persona_corpus, tokenizer)

    # Build training data
    print(f"\nBuilding v2 training data (placeholder='{PLACEHOLDER_TOKEN}')...")
    training_data = build_training_mixture(
        args.corpus, args.conv_qa, args.persona_corpus,
        tokenizer, args.model, layer_percents, task_sizes,
    )

    if not training_data:
        print("ERROR: No training data generated!")
        return

    # Split off 100 random examples per task as eval
    by_type = defaultdict(list)
    for dp in training_data:
        by_type[dp.datapoint_type].append(dp)

    eval_datasets = {}
    final_training = []
    eval_rng = random.Random(42)
    for dtype, dps in by_type.items():
        if len(dps) > 100:
            eval_rng.shuffle(dps)
            eval_datasets[dtype] = dps[:100]
            final_training.extend(dps[100:])
        else:
            final_training.extend(dps)

    print(f"\nTraining: {len(final_training)}, Eval: {sum(len(v) for v in eval_datasets.values())}")
    for name, items in eval_datasets.items():
        print(f"  eval/{name}: {len(items)} items")

    # Download AO checkpoint
    ao_checkpoints = {
        "Qwen/Qwen3-1.7B": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B",
        "Qwen/Qwen3-8B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
    }

    lora_local_path = None
    hf_repo = ao_checkpoints.get(args.model)
    if hf_repo:
        from huggingface_hub import snapshot_download
        lora_local_path = snapshot_download(hf_repo)
        print(f"AO checkpoint downloaded to: {lora_local_path}")

    cfg = SelfInterpTrainingConfig(
        model_name=args.model,
        hook_onto_layer=1,
        layer_percents=layer_percents,
        steering_coefficient=1.0,
        lr=args.lr,
        num_epochs=args.epochs,
        train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_dir=args.save_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run or f"cot_oracle_v2_{args.model.split('/')[-1]}",
        gradient_checkpointing=args.gradient_checkpointing,
        load_lora_path=lora_local_path,
        eval_on_start=True,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Initialize distributed (required by AO's train_model)
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Install hooks (no multi-layer materialization needed in v2!)
    if not args.no_unfaith_evals and Path(args.eval_dir).exists():
        install_unfaithfulness_eval_hook(
            model_name=args.model,
            eval_dir=args.eval_dir,
            fast_n=args.fast_eval_n,
        )

    install_per_task_loss_hook()
    install_fuzzy_eval_hook()

    # Login to wandb
    import os
    assert os.environ.get("WANDB_API_KEY"), "Set WANDB_API_KEY env var"
    import wandb

    # Shuffle training data
    random.seed(42)
    random.shuffle(final_training)
    print(f"Shuffled {len(final_training)} training examples")

    print(f"\nStarting training:")
    print(f"  Model: {cfg.model_name}")
    print(f"  AO checkpoint: {cfg.load_lora_path}")
    print(f"  Placeholder: '{PLACEHOLDER_TOKEN}'")
    print(f"  LR: {cfg.lr}")
    print(f"  Batch size: {cfg.train_batch_size}")
    print(f"  Epochs: {cfg.num_epochs}")
    print(f"  Total steps: ~{len(final_training) // cfg.train_batch_size}")
    print(f"  Save dir: {cfg.save_dir}")
    print(f"  Tasks: {sorted(set(dp.datapoint_type for dp in final_training))}")

    train_model(
        cfg=cfg,
        training_data=final_training,
        eval_datasets=eval_datasets,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        model_kwargs={"attn_implementation": "sdpa"},
    )


if __name__ == "__main__":
    main()

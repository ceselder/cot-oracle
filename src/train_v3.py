"""
Train CoT Oracle v3: From Scratch + FineWeb + Fixed Evals

Changes from v2:
  - Train from SCRATCH (fresh LoRA) — no Adam's AO checkpoint
  - FineWeb context prediction in mix (PastLens-style, HF streaming)
  - Fuzzy eval scoring for generation tasks (token F1, substring match)
  - Unfaithfulness evals ON by default
  - Harder evals: rot13, logical_leaps, held_out_reconstruction, fineweb_context_pred
  - Medium corpus support (47K entries vs 1K mini)

Task mix (~425K examples):
  1. CoT context prediction (100K) — random positions in CoT
  2. FineWeb context prediction (100K) — PastLens from web text (streaming)
  3. Answer prediction (20K) — predict final answer from prefix activations
  4. Full reconstruction (15K) — predict entire CoT from boundary activations
  5. Causal prediction (30K) — autoregressive: feed 1..i, predict next chunk
  6. Conversational QA (5K) — 107 question types, Gemini Flash responses
  7. Decorative CoT (10K) — load-bearing vs decorative binary
  8. Domain classification (15K) — multi-class
  9. Correctness prediction (15K) — binary correct/incorrect
  10. Persona detection (15K) — multi-class unverbalized prompt

Usage:
    torchrun --nproc_per_node=1 src/train_v3.py \
        --corpus data/cot_corpus_v5/corpus_medium.jsonl \
        --conv-qa data/hf_upload/cot_conversational_qa.jsonl \
        --model Qwen/Qwen3-8B
"""

import argparse
import logging
import random
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

# Suppress verbose warnings that spam the log during generation
warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

from core.ao_repo import ensure_ao_repo_on_path

ensure_ao_repo_on_path()

import torch

import nl_probes.utils.dataset_utils as du_module
from nl_probes.utils.dataset_utils import (
    create_training_datapoint,
    TrainingDataPoint,
)
import nl_probes.sft as sft_module
from nl_probes.sft import train_model
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.utils.common import load_tokenizer

# ── Override placeholder token ──
PLACEHOLDER_TOKEN = " ¶"
du_module.SPECIAL_TOKEN = PLACEHOLDER_TOKEN

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
from dataset_classes.fineweb_context_prediction import load_fineweb_context_prediction_data

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
    """Convert dataset loader output to AO TrainingDataPoint objects."""
    training_data = []
    skipped = 0

    for item in raw_data:
        try:
            layer = item.get("layer")
            if layer is None:
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
    """Build the v3 mixed training data (CoT tasks + FineWeb)."""

    all_data = []

    # Task 1: CoT Context Prediction
    print("\n=== Task 1: CoT Context Prediction ===")
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

    # Task 2: FineWeb Context Prediction (streaming)
    n_fineweb = task_sizes.get("fineweb", 100000)
    if n_fineweb > 0:
        print(f"\n=== Task 2: FineWeb Context Prediction ({n_fineweb}) ===")
        try:
            raw = load_fineweb_context_prediction_data(
                tokenizer, model_name, layer_percents,
                num_examples=n_fineweb,
            )
            data = dicts_to_training_data(raw, tokenizer)
            print(f"  -> {len(data)} examples")
            all_data.extend(data)
        except Exception as e:
            print(f"  FAILED: {e}")

    # Task 3: Answer Prediction
    print("\n=== Task 3: Answer Prediction ===")
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

    # Task 4: Full CoT Reconstruction
    print("\n=== Task 4: Full CoT Reconstruction ===")
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

    # Task 5: Causal Prediction
    print("\n=== Task 5: Causal Prediction ===")
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

    # Task 6: Conversational QA
    if conv_qa_path and Path(conv_qa_path).exists():
        print("\n=== Task 6: Conversational QA ===")
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
        print(f"\n  Skipping Task 6 (no conversational QA at {conv_qa_path})")

    # Task 7: Decorative CoT
    print("\n=== Task 7: Decorative CoT ===")
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

    # Task 8: Domain Classification
    print("\n=== Task 8: Domain Classification ===")
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

    # Task 9: Correctness Prediction
    print("\n=== Task 9: Correctness Prediction ===")
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

    # Task 10: Persona Detection
    if persona_corpus_path and Path(persona_corpus_path).exists():
        print("\n=== Task 10: Persona Detection ===")
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
        print(f"\n  Skipping Task 10 (no persona corpus at {persona_corpus_path})")

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

    CLASSIFICATION_TASKS = {
        "cot_decorative", "cot_domain", "cot_correctness", "cot_persona",
    }
    GENERATION_TASKS = {
        "cot_full_reconstruction", "cot_causal_prediction",
        "cot_conversational", "cot_context_prediction",
        "fineweb_context_prediction",
    }
    ANSWER_TASKS = {"cot_answer_prediction"}

    def token_f1(prediction: str, reference: str) -> float:
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
        scores = []
        samples = []

        for resp, dp in zip(responses, eval_data):
            pred = resp.api_response.strip()
            target = dp.target_output.strip()

            if ds_name in CLASSIFICATION_TASKS:
                pred_lower = pred.lower()
                target_lower = target.lower()
                score = 1.0 if target_lower in pred_lower else 0.0
            elif ds_name in ANSWER_TASKS:
                pred_clean = parse_answer(pred)
                target_clean = parse_answer(target)
                score = 1.0 if pred_clean == target_clean else 0.0
            else:
                score = token_f1(pred, target)

            scores.append(score)
            if len(samples) < 3:
                samples.append((pred[:200], target[:200], score))

        avg = sum(scores) / len(scores) if scores else 0.0
        return avg, samples

    _original_eval = sft_module.eval_all_datasets

    def patched_eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step):
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

            # Always show 1 sample for quick inspection
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
            with torch.no_grad():
                logits = outputs.logits.detach()
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


def install_unfaithfulness_eval_hook(model_name, eval_dir="data/evals", fast_n=5,
                                     unfaith_every=2000):
    """Monkey-patch AO's eval to run unfaithfulness evals alongside training evals."""
    import wandb
    from evals.common import load_eval_items
    from evals.score_oracle import score_eval, EVAL_PARSING
    from evals.run_evals import run_single_item, set_oracle_mode

    # Configure oracle mode for trained oracle (¶ tokens, stride positions)
    # During training, the adapter being trained is "default"
    set_oracle_mode(trained=True, oracle_adapter_name="default")

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
    print(f"Unfaithfulness eval hook: {len(fast_items)} evals, {total_items} total items (every {unfaith_every} steps)")

    _original_eval = sft_module.eval_all_datasets

    def patched_eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step):
        _original_eval(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step)

        if global_step % unfaith_every != 0:
            print(f"  (Skipping unfaith evals at step {global_step}, next at {((global_step // unfaith_every) + 1) * unfaith_every})")
            return

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
    print("Installed unfaithfulness eval hook (trained oracle mode)")


def main():
    parser = argparse.ArgumentParser(description="Train CoT Oracle v3")
    parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--conv-qa", default=None, help="Path to conversational QA JSONL")
    parser.add_argument("--persona-corpus", default=None, help="Path to persona corpus.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-dir", default="checkpoints/cot_oracle_v3")
    parser.add_argument("--wandb-project", default="cot_oracle")
    parser.add_argument("--wandb-run", default="")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.add_argument("--eval-dir", default="data/evals")
    parser.add_argument("--fast-eval-n", type=int, default=10)
    parser.add_argument("--no-unfaith-evals", action="store_true")

    # v3: Option to continue from AO checkpoint (default: fresh LoRA)
    parser.add_argument("--from-ao-checkpoint", action="store_true",
                        help="Continue from Adam's AO checkpoint (default: fresh LoRA)")

    # Task size overrides
    parser.add_argument("--n-context-pred", type=int, default=100000)
    parser.add_argument("--n-fineweb", type=int, default=100000)
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
        "fineweb": args.n_fineweb,
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
    print(f"\nBuilding v3 training data (placeholder='{PLACEHOLDER_TOKEN}')...")
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

    # v3: Fresh LoRA by default, optionally continue from AO checkpoint
    lora_local_path = None
    if args.from_ao_checkpoint:
        ao_checkpoints = {
            "Qwen/Qwen3-1.7B": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B",
            "Qwen/Qwen3-8B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
        }
        hf_repo = ao_checkpoints.get(args.model)
        if hf_repo:
            from huggingface_hub import snapshot_download
            lora_local_path = snapshot_download(hf_repo)
            print(f"AO checkpoint downloaded to: {lora_local_path}")
    else:
        print("Training from scratch (fresh LoRA, no AO checkpoint)")

    cfg = SelfInterpTrainingConfig(
        model_name=args.model,
        hook_onto_layer=1,
        layer_percents=layer_percents,
        steering_coefficient=1.0,
        lr=args.lr,
        num_epochs=args.epochs,
        train_batch_size=args.batch_size,
        eval_batch_size=4,  # Must be small — materialization OOMs at larger batches (CUDA driver 525 reports as "invalid argument")
        gradient_accumulation_steps=1,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_dir=args.save_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run or f"cot_oracle_v3_{args.model.split('/')[-1]}",
        gradient_checkpointing=args.gradient_checkpointing,
        load_lora_path=lora_local_path,
        eval_on_start=False,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Initialize distributed (required by AO's train_model)
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Install hooks — order matters: fuzzy first, then unfaith wraps it
    # (unfaith hook chains to _original_eval; fuzzy hook replaces entirely)
    install_per_task_loss_hook()
    install_fuzzy_eval_hook()

    if not args.no_unfaith_evals and Path(args.eval_dir).exists():
        install_unfaithfulness_eval_hook(
            model_name=args.model,
            eval_dir=args.eval_dir,
            fast_n=args.fast_eval_n,
            unfaith_every=2000,
        )

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
    print(f"  LoRA: {'from AO checkpoint' if lora_local_path else 'FRESH (from scratch)'}")
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

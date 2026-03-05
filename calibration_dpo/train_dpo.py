"""Calibration DPO training loop.

Pipeline: sample CoT → extract activations → generate rollouts → judge →
construct DPO pairs → train step.

Async overlap: while Gemini judges batch N, GPU generates rollouts for batch N+1.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "ao_reference"))
sys.path.insert(0, str(project_root / "calibration_dpo"))

from core.ao import TRAINED_PLACEHOLDER, get_hf_submodule

from activations import extract_activations
from data_sampler import CoTSampler
from dpo_loss import DPOBatchItem, SFTItem, compute_dpo_loss, compute_sft_loss
from judge import JudgeResult, RolloutRating, judge_batch
from prompts import REFUSAL_PARAPHRASES, SPECIFICITY_SUFFIX, sample_prompt, sample_refusal
from rollouts import generate_rollouts, _build_manual_prefix_token_ids, _build_oracle_prefix


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict, device: torch.device) -> tuple[PeftModel, AutoTokenizer]:
    """Load base model + dual adapters (policy=trainable, reference=frozen)."""
    model_cfg = cfg["model"]
    dtype = torch.bfloat16

    # Prefer flash_attention_2 but fall back to sdpa
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    print(f"Loading base model: {model_cfg['base']} (attn={attn_impl})")
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["base"],
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation=attn_impl,
    )
    base.enable_input_require_grads()
    base.use_cache = False
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )

    print(f"Loading checkpoint: {model_cfg['checkpoint']}")
    model = PeftModel.from_pretrained(
        base, model_cfg["checkpoint"],
        adapter_name="policy",
        is_trainable=True,
        autocast_adapter_dtype=False,
    )

    # Load same checkpoint as frozen reference adapter
    model.load_adapter(model_cfg["checkpoint"], adapter_name="reference")
    for n, p in model.named_parameters():
        if "reference" in n:
            p.requires_grad = False

    model.set_adapter("policy")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    return model, tokenizer


def build_oracle_input_ids(
    tokenizer,
    activations: torch.Tensor,
    oracle_prompt: str,
    layers: list[int],
) -> tuple[list[int], list[int]]:
    """Build tokenized oracle input with placeholder positions.

    Returns (input_ids, ph_positions).
    """
    ph_token = TRAINED_PLACEHOLDER
    ph_id_list = tokenizer.encode(ph_token, add_special_tokens=False)
    assert len(ph_id_list) == 1
    ph_id = ph_id_list[0]

    num_positions = activations.shape[0]
    prefix = _build_oracle_prefix(num_positions, layers, ph_token)
    full_prompt = prefix + oracle_prompt

    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    prefix_idx = formatted.find(prefix)
    assert prefix_idx >= 0, "Prefix not found in formatted text"
    before_ids = tokenizer.encode(formatted[:prefix_idx], add_special_tokens=False)
    after_ids = tokenizer.encode(formatted[prefix_idx + len(prefix):], add_special_tokens=False)
    prefix_ids, rel_positions = _build_manual_prefix_token_ids(
        tokenizer, num_positions, layers, ph_id,
    )

    input_ids = before_ids + prefix_ids + after_ids
    positions = [len(before_ids) + p for p in rel_positions]
    return input_ids, positions


def build_dpo_items(
    tokenizer,
    activations: torch.Tensor,
    oracle_prompt: str,
    layers: list[int],
    rollouts: list[str],
    judge_result: JudgeResult,
    rng: random.Random,
    max_refusal_pairs: int = 3,
    refusal_sft_weight: float = 3.0,
) -> tuple[list[DPOBatchItem], list[SFTItem]]:
    """Construct DPO pairs and SFT targets from judge verdicts.

    Returns (dpo_items, sft_items).
    """
    prompt_ids, ph_positions = build_oracle_input_ids(
        tokenizer, activations, oracle_prompt, layers,
    )

    def _make_full_ids(response_text: str) -> tuple[list[int], list[int]]:
        """Build full input_ids and labels for a response."""
        resp_ids = tokenizer.encode(response_text, add_special_tokens=False)
        # Add EOS
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            resp_ids = resp_ids + [eos_id]
        full_ids = prompt_ids + resp_ids
        # Labels: -100 for prompt, actual ids for response
        labels = [-100] * len(prompt_ids) + resp_ids
        return full_ids, labels

    dpo_items = []
    sft_items = []

    be_specific = oracle_prompt.endswith(SPECIFICITY_SUFFIX)

    # Map ratings by index
    rating_map: dict[int, RolloutRating] = {}
    for r in judge_result.ratings:
        rating_map[r.index] = r

    all_bad = all(
        rating_map.get(i + 1, RolloutRating(index=i+1, rating="bad")).rating == "bad"
        for i in range(len(rollouts))
    )

    # If ALL rollouts are bad, skip DPO entirely — just do SFT on refusal
    # (unless "be specific" mode — then still SFT on ideal, skip refusal SFT)
    if not all_bad:
        for i, rollout_text in enumerate(rollouts):
            rating = rating_map.get(i + 1)
            if rating is None:
                continue

            if rating.rating == "indeterminate":
                continue

            refusal = sample_refusal(rng)

            if rating.rating == "good":
                # If malformed, prefer reformatted over original
                if rating.malformed and rating.reformatted:
                    ref_chosen_ids, ref_chosen_labels = _make_full_ids(rating.reformatted)
                    ref_rejected_ids, ref_rejected_labels = _make_full_ids(rollout_text)
                    dpo_items.append(DPOBatchItem(
                        chosen_ids=ref_chosen_ids,
                        rejected_ids=ref_rejected_ids,
                        chosen_labels=ref_chosen_labels,
                        rejected_labels=ref_rejected_labels,
                        activations=activations,
                        ph_positions=ph_positions,
                        label="reformatted>malformed",
                    ))
                # No model>refusal DPO — we don't want to push refusal
                # probability down. SFT on ideal handles positive signal.

            elif rating.rating == "mixed":
                # Disabled: corrections inject judge's text-based interpretation
                # which may not match what's actually in activations.
                # SFT on ideal (synthesized from model's own outputs) handles this.
                ENABLE_MIXED_DPO = True
                if ENABLE_MIXED_DPO and rating.correction:
                    chosen_text = rating.correction
                    rejected_text = rollout_text
                    if rating.malformed and rating.reformatted:
                        chosen_text = rating.reformatted
                    chosen_ids, chosen_labels = _make_full_ids(chosen_text)
                    rejected_ids, rejected_labels = _make_full_ids(rejected_text)
                    dpo_items.append(DPOBatchItem(
                        chosen_ids=chosen_ids,
                        rejected_ids=rejected_ids,
                        chosen_labels=chosen_labels,
                        rejected_labels=rejected_labels,
                        activations=activations,
                        ph_positions=ph_positions,
                        label="correction>model",
                    ))

            elif rating.rating == "bad":
                # Skip refusal pairs in "be specific" mode — we asked for detail
                if be_specific:
                    continue
                # Refusal > bad rollout (capped to avoid overwhelming signal)
                n_refusal_pairs = sum(1 for d in dpo_items if d.label == "refusal>model")
                if n_refusal_pairs >= max_refusal_pairs:
                    continue
                chosen_ids, chosen_labels = _make_full_ids(refusal)
                rejected_ids, rejected_labels = _make_full_ids(rollout_text)
                dpo_items.append(DPOBatchItem(
                    chosen_ids=chosen_ids,
                    rejected_ids=rejected_ids,
                    chosen_labels=chosen_labels,
                    rejected_labels=rejected_labels,
                    activations=activations,
                    ph_positions=ph_positions,
                    label="refusal>model",
                ))

    # Specificity DPO: pair specific good rollouts against vague good rollouts
    specific_good = [
        (i, rollouts[i]) for i in range(len(rollouts))
        if rating_map.get(i + 1) and rating_map[i + 1].rating == "good" and not rating_map[i + 1].vague
    ]
    vague_good = [
        (i, rollouts[i]) for i in range(len(rollouts))
        if rating_map.get(i + 1) and rating_map[i + 1].rating == "good" and rating_map[i + 1].vague
    ]
    for _, specific_text in specific_good:
        for _, vague_text in vague_good:
            chosen_ids, chosen_labels = _make_full_ids(specific_text)
            rejected_ids, rejected_labels = _make_full_ids(vague_text)
            dpo_items.append(DPOBatchItem(
                chosen_ids=chosen_ids,
                rejected_ids=rejected_ids,
                chosen_labels=chosen_labels,
                rejected_labels=rejected_labels,
                activations=activations,
                ph_positions=ph_positions,
                label="specific>vague",
            ))

    # SFT on ideal response (always)
    ideal = judge_result.ideal_response
    if ideal is None and all_bad:
        ideal = sample_refusal(rng)

    if ideal:
        ideal_ids, ideal_labels = _make_full_ids(ideal)
        weight = 1.0  # refusal_sft_weight applied externally if all_bad
        sft_items.append(SFTItem(
            input_ids=ideal_ids,
            labels=ideal_labels,
            activations=activations,
            ph_positions=ph_positions,
            weight=weight,
        ))

    # Extra refusal SFT if all bad (skip in "be specific" mode)
    if all_bad and not be_specific:
        refusal = sample_refusal(rng)
        ref_ids, ref_labels = _make_full_ids(refusal)
        sft_items.append(SFTItem(
            input_ids=ref_ids,
            labels=ref_labels,
            activations=activations,
            ph_positions=ph_positions,
            weight=refusal_sft_weight,
        ))

    return dpo_items, sft_items


def _run_judge_async(
    examples, rollouts_per_example, oracle_prompts, tokenizer, judge_cfg,
) -> list[JudgeResult]:
    """Run async judge in a new event loop (for use from sync code / threads)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            judge_batch(
                examples=examples,
                rollouts_per_example=rollouts_per_example,
                oracle_prompts=oracle_prompts,
                tokenizer=tokenizer,
                model=judge_cfg["model"],
                temperature=judge_cfg["temperature"],
                timeout_sec=judge_cfg["timeout_sec"],
                max_concurrent=judge_cfg["max_concurrent"],
            )
        )
    finally:
        loop.close()


def train(cfg: dict) -> None:
    device = torch.device("cuda")
    rng = random.Random(42)

    # Load model
    model, tokenizer = load_model(cfg, device)
    layers = cfg["model"]["act_layers"]
    injection_layer = cfg["model"]["injection_layer"]

    # Data sampler
    sampler = CoTSampler(
        corpus_name=cfg["data"]["corpus"],
        tokenizer=tokenizer,
        layers=layers,
        stride=cfg["data"]["stride"],
        min_positions=cfg["data"]["min_positions"],
        max_positions=cfg["data"]["max_positions"],
    )

    # Optimizer
    train_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg["lr"],
        weight_decay=0.01,
    )

    # LR scheduler with warmup
    warmup_steps = train_cfg["warmup_steps"]
    max_steps = train_cfg["max_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0  # constant after warmup

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Wandb
    wandb_cfg = cfg.get("wandb", {})
    try:
        import wandb
        wandb.login()
        wandb.init(
            project=wandb_cfg.get("project", "cot-oracle-calibration-dpo"),
            config=cfg,
        )
        use_wandb = True
        print(f"Wandb run: {wandb.run.name} ({wandb.run.id})")
    except Exception as e:
        print(f"Wandb init failed: {e}, continuing without logging")
        use_wandb = False

    log_cfg = cfg.get("logging", {})
    console_every = log_cfg.get("console_every", 10)
    table_every = log_cfg.get("table_every", 50)
    save_every = train_cfg.get("save_every", 200)

    dpo_cfg = cfg["dpo"]
    rollout_cfg = cfg["rollouts"]
    judge_cfg = cfg["judge"]

    batch_size = train_cfg["batch_size"]
    grad_accum = train_cfg["gradient_accumulation_steps"]

    # --- Async pipeline ---
    # Queue for (examples, rollouts, oracle_prompts) waiting for judge results
    judge_queue: Queue = Queue(maxsize=2)
    result_queue: Queue = Queue(maxsize=2)

    def judge_worker():
        """Background thread that sends batches to Gemini for judging."""
        while True:
            item = judge_queue.get()
            if item is None:
                break
            examples, rollouts_list, oracle_prompts, acts_list = item
            t0 = time.time()
            results = _run_judge_async(
                examples, rollouts_list, oracle_prompts, tokenizer, judge_cfg,
            )
            judge_time = time.time() - t0
            result_queue.put((examples, rollouts_list, oracle_prompts, acts_list, results, judge_time))

    judge_thread = Thread(target=judge_worker, daemon=True)
    judge_thread.start()

    # --- Training loop ---
    global_step = 0
    accum_dpo_items: list[DPOBatchItem] = []
    accum_sft_items: list[SFTItem] = []
    metrics_accum = defaultdict(list)

    # Async pipeline state
    pending_judge = False
    prev_examples = prev_rollouts = prev_prompts = judge_results = None

    print(f"\nStarting training: {max_steps} steps, batch={batch_size}, accum={grad_accum}")
    print(f"Rollouts: {rollout_cfg['n']}× at T={rollout_cfg['temperature']}")
    print(f"DPO beta={dpo_cfg['beta']}, sft_weight={dpo_cfg['sft_weight']}")

    try:
        while global_step < max_steps:
            step_t0 = time.time()

            # Sample batch
            examples = sampler.sample_batch(batch_size)
            oracle_prompts = [sample_prompt(rng) for _ in range(batch_size)]

            # Extract activations (adapter disabled)
            activations_list = extract_activations(
                model, tokenizer, examples, layers, device,
            )

            # Generate rollouts (adapter enabled)
            gpu_t0 = time.time()
            rollouts_list = generate_rollouts(
                model=model,
                tokenizer=tokenizer,
                activations_list=activations_list,
                oracle_prompts=oracle_prompts,
                layers=layers,
                n_rollouts=rollout_cfg["n"],
                temperature=rollout_cfg["temperature"],
                max_new_tokens=rollout_cfg["max_new_tokens"],
                generation_batch_size=rollout_cfg["generation_batch_size"],
                injection_layer=injection_layer,
                adapter_name="policy",
                device=device,
            )
            gpu_time = time.time() - gpu_t0

            # Submit to judge (async) — include activations so they're available when results come back
            judge_queue.put((examples, rollouts_list, oracle_prompts, activations_list))

            # If we have pending results from previous batch, process them
            if pending_judge:
                prev_examples, prev_rollouts, prev_prompts, prev_acts, judge_results, judge_time = result_queue.get()

                # Build DPO pairs
                first_ex_dpo = []
                first_ex_sft = []
                for i in range(len(prev_examples)):
                    if judge_results[i] is None or not judge_results[i].ratings:
                        continue
                    dpo_items, sft_items = build_dpo_items(
                        tokenizer=tokenizer,
                        activations=prev_acts[i],
                        oracle_prompt=prev_prompts[i],
                        layers=layers,
                        rollouts=prev_rollouts[i],
                        judge_result=judge_results[i],
                        rng=rng,
                        max_refusal_pairs=dpo_cfg.get("max_refusal_pairs", 3),
                        refusal_sft_weight=dpo_cfg.get("refusal_sft_weight", 3.0),
                    )
                    if i == 0:
                        first_ex_dpo = list(dpo_items)
                        first_ex_sft = list(sft_items)
                    accum_dpo_items.extend(dpo_items)
                    accum_sft_items.extend(sft_items)

                # Compute rating fractions
                all_ratings = []
                for jr in judge_results:
                    if jr and jr.ratings:
                        all_ratings.extend(r.rating for r in jr.ratings)
                if all_ratings:
                    total = len(all_ratings)
                    metrics_accum["good_frac"].append(sum(1 for r in all_ratings if r == "good") / total)
                    metrics_accum["bad_frac"].append(sum(1 for r in all_ratings if r == "bad") / total)
                    metrics_accum["mixed_frac"].append(sum(1 for r in all_ratings if r == "mixed") / total)
                    metrics_accum["indeterminate_frac"].append(sum(1 for r in all_ratings if r == "indeterminate") / total)

                # Vague fraction (over all rated rollouts)
                all_vague = []
                for jr in judge_results:
                    if jr and jr.ratings:
                        all_vague.extend(r.vague for r in jr.ratings)
                if all_vague:
                    metrics_accum["vague_frac"].append(sum(all_vague) / len(all_vague))

                # Exact refusal fraction (over raw rollouts)
                refusal_set = set(REFUSAL_PARAPHRASES)
                all_rollout_texts = [r for rs in prev_rollouts for r in rs]
                if all_rollout_texts:
                    n_exact_refusal = sum(1 for r in all_rollout_texts if r.strip() in refusal_set)
                    metrics_accum["exact_refusal_frac"].append(n_exact_refusal / len(all_rollout_texts))

                # Check all-bad fraction
                n_all_bad = 0
                for jr in judge_results:
                    if jr and jr.ratings:
                        if all(r.rating == "bad" for r in jr.ratings):
                            n_all_bad += 1
                if judge_results:
                    metrics_accum["all_bad_frac"].append(n_all_bad / len(judge_results))

                metrics_accum["judge_time_s"].append(judge_time)

                # Detailed per-rollout log (first example only, every table_every steps)
                if global_step % table_every < batch_size and prev_examples:
                    ex = prev_examples[0]
                    jr = judge_results[0]
                    if jr and jr.ratings:
                        print(f"\n{'═'*70}")
                        print(f"  EXAMPLE DETAIL")
                        print(f"{'═'*70}")
                        print(f"  Question:  {ex['question'][:120]}...")
                        print(f"  CoT:       {ex['cot_response'][:120]}...")
                        print(f"  Prompt:    {prev_prompts[0]}")
                        print(f"{'─'*70}")
                        # Show DPO pairs with rating labels
                        if first_ex_dpo:
                            print(f"  DPO PAIRS ({len(first_ex_dpo)}):")
                            for pi, pair in enumerate(first_ex_dpo):
                                chosen_text = tokenizer.decode(
                                    [t for t, l in zip(pair.chosen_ids, pair.chosen_labels) if l != -100],
                                    skip_special_tokens=True,
                                )[:150].replace('\n', ' ')
                                rejected_text = tokenizer.decode(
                                    [t for t, l in zip(pair.rejected_ids, pair.rejected_labels) if l != -100],
                                    skip_special_tokens=True,
                                )[:150].replace('\n', ' ')
                                # Label shows pair type and which side is model-generated
                                tag = pair.label or "?"
                                print(f"  {pi+1}. [{tag}]")
                                c_src = "model" if tag in ("model>refusal", "specific>vague") else "judge"
                                r_src = "model" if tag in ("refusal>model", "correction>model", "reformatted>malformed") else "judge"
                                print(f"     ✓ ({c_src}) {chosen_text}")
                                print(f"     ✗ ({r_src}) {rejected_text}")
                        elif first_ex_sft:
                            print(f"  ALL BAD — SFT only:")
                            for si, sft in enumerate(first_ex_sft):
                                sft_text = tokenizer.decode(
                                    [t for t, l in zip(sft.input_ids, sft.labels) if l != -100],
                                    skip_special_tokens=True,
                                )[:150].replace('\n', ' ')
                                print(f"  SFT (w={sft.weight:.1f}): {sft_text}")
                        if jr.ideal_response:
                            print(f"{'─'*70}")
                            ideal_trunc = jr.ideal_response[:200].replace('\n', ' ')
                            print(f"  IDEAL: {ideal_trunc}")
                        print(f"{'═'*70}\n")

            pending_judge = True

            # If we have enough accumulated items, do a training step
            if len(accum_dpo_items) >= grad_accum or len(accum_sft_items) >= grad_accum:
                model.train()
                model.set_adapter("policy")

                # Take a subset for this step
                step_dpo = accum_dpo_items[:grad_accum * 2]
                step_sft = accum_sft_items[:grad_accum]
                accum_dpo_items = accum_dpo_items[grad_accum * 2:]
                accum_sft_items = accum_sft_items[grad_accum:]

                optimizer.zero_grad()

                # DPO loss
                dpo_loss, dpo_metrics = compute_dpo_loss(
                    model, step_dpo, injection_layer, device,
                    beta=dpo_cfg["beta"],
                )

                # SFT loss
                sft_loss, sft_metrics = compute_sft_loss(
                    model, step_sft, injection_layer, device,
                )

                dpo_lr_scale = dpo_cfg.get("lr_scale", 1.0)
                total_loss = dpo_lr_scale * dpo_loss + dpo_cfg["sft_weight"] * sft_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train_cfg["max_grad_norm"],
                )
                optimizer.step()
                scheduler.step()

                global_step += 1
                step_time = time.time() - step_t0

                metrics_accum["dpo_loss"].append(dpo_metrics["dpo_loss"])
                metrics_accum["sft_loss"].append(sft_metrics["sft_loss"])
                metrics_accum["total_loss"].append(total_loss.item())
                metrics_accum["reward_margin"].append(dpo_metrics["reward_margin"])
                metrics_accum["gpu_time_s"].append(gpu_time)

                # Console + wandb logging
                if global_step % console_every == 0:
                    avg = {k: sum(v) / len(v) for k, v in metrics_accum.items() if v}
                    print(f"\n{'─'*60}")
                    print(f"  Step {global_step}/{max_steps}")
                    print(f"{'─'*60}")
                    print(f"  Loss     dpo: {avg.get('dpo_loss', 0):.4f}   sft: {avg.get('sft_loss', 0):.4f}   total: {avg.get('total_loss', 0):.4f}")
                    print(f"  Margin   {avg.get('reward_margin', 0):+.3f}")
                    print(f"  Ratings  good: {avg.get('good_frac', 0):.1%}  bad: {avg.get('bad_frac', 0):.1%}  mixed: {avg.get('mixed_frac', 0):.1%}  vague: {avg.get('vague_frac', 0):.1%}")
                    print(f"  Refusal  exact: {avg.get('exact_refusal_frac', 0):.1%}")
                    print(f"  Timing   gpu: {avg.get('gpu_time_s', 0):.1f}s  judge: {avg.get('judge_time_s', 0):.1f}s")

                    if use_wandb:
                        log_dict = {f"train/{k}": v for k, v in avg.items()}
                        log_dict["train/lr"] = scheduler.get_last_lr()[0]
                        wandb.log(log_dict, step=global_step)

                    metrics_accum.clear()

                # Table logging
                if use_wandb and global_step % table_every == 0 and prev_examples:
                    try:
                        table = wandb.Table(columns=[
                            "step", "cot_excerpt", "prompt",
                            "rollouts", "ratings", "chosen", "rejected", "ideal",
                        ])
                        for i in range(min(len(prev_examples), 3)):
                            ex = prev_examples[i]
                            jr = judge_results[i] if judge_results[i] else None
                            cot_excerpt = ex["cot_response"][:300] + "..."
                            rollout_strs = "\n---\n".join(
                                f"R{j+1}: {r[:200]}" for j, r in enumerate(prev_rollouts[i])
                            )
                            rating_strs = ""
                            chosen_str = ""
                            rejected_str = ""
                            ideal_str = ""
                            if jr:
                                rating_strs = ", ".join(
                                    f"R{r.index}={r.rating}" for r in jr.ratings
                                )
                                ideal_str = jr.ideal_response or ""
                            table.add_data(
                                global_step, cot_excerpt, prev_prompts[i],
                                rollout_strs, rating_strs, chosen_str, rejected_str, ideal_str,
                            )
                        wandb.log({"train/examples": table}, step=global_step)
                    except Exception as e:
                        print(f"  [wandb] Table logging failed: {e}")

                # Save checkpoint + upload to HF
                if global_step % save_every == 0:
                    save_dir = Path(f"checkpoints/calibration_dpo_step_{global_step}")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(save_dir))
                    tokenizer.save_pretrained(str(save_dir))
                    print(f"  Saved checkpoint: {save_dir}")
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi()
                        repo_id = "ceselder/cot-oracle-calibration-dpo"
                        api.create_repo(repo_id, exist_ok=True)
                        api.upload_folder(
                            folder_path=str(save_dir),
                            path_in_repo=f"step_{global_step}",
                            repo_id=repo_id,
                        )
                        print(f"  Uploaded to HF: {repo_id}/step_{global_step}")
                    except Exception as e:
                        print(f"  HF upload failed: {e}")

                # Clean up
                del dpo_loss, sft_loss, total_loss
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Stop judge thread
        judge_queue.put(None)
        judge_thread.join(timeout=5)

        # Final save
        if global_step > 0:
            save_dir = Path(f"checkpoints/calibration_dpo_final_step_{global_step}")
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            print(f"Final checkpoint: {save_dir}")

            # Upload to HF
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                repo_id = "ceselder/cot-oracle-calibration-dpo"
                api.create_repo(repo_id, exist_ok=True)
                api.upload_folder(
                    folder_path=str(save_dir),
                    path_in_repo=f"step_{global_step}_final",
                    repo_id=repo_id,
                )
                print(f"Uploaded to HF: {repo_id}/step_{global_step}_final")
            except Exception as e:
                print(f"HF upload failed: {e}")

        if use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Calibration DPO training")
    parser.add_argument("--config", type=str, default="calibration_dpo/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GRPO training for the CoT Oracle.

Online loop: sample CoT → extract activations → generate rollouts →
score with rubric judge → compute group advantages → policy gradient.

Usage:
    cd /root/cot-oracle
    export HF_TOKEN=... OPENROUTER_API_KEY=... WANDB_API_KEY=...
    python calibration_grpo/train_grpo.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ao_reference"))
sys.path.insert(0, str(ROOT / "calibration_dpo"))  # reuse activations, rollouts, data_sampler
sys.path.insert(0, str(ROOT / "calibration_grpo"))

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from core.ao import (
    TRAINED_PLACEHOLDER,
    choose_attn_implementation,
    get_hf_submodule,
    get_steering_hook,
    add_hook,
    using_adapter,
)
from nl_probes.utils.activation_utils import collect_activations_multiple_layers

# Reuse from calibration_dpo
from activations import extract_activations
from data_sampler import CoTSampler
from rollouts import generate_rollouts, build_oracle_input_ids

# GRPO modules
from grpo_loss import GRPOItem, compute_grpo_loss, compute_old_logprobs
from judge import judge_batch
from reward import CRITERIA_NAMES, RubricResult, compute_group_advantages, compute_rewards


def load_config(path: str = None) -> dict:
    if path is None:
        path = str(Path(__file__).parent / "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict, device: str = "cuda") -> tuple[PeftModel, AutoTokenizer]:
    """Load base model + policy LoRA. Single adapter (no reference needed for DR-GRPO)."""
    model_name = cfg["model"]["base"]
    checkpoint = cfg["model"]["checkpoint"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation=choose_attn_implementation(model_name),
    )
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    model.print_trainable_parameters()
    return model, tokenizer


def train(cfg: dict):
    device = "cuda"
    tcfg = cfg["training"]
    rcfg = cfg["rollouts"]
    gcfg = cfg["grpo"]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY")

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(cfg, device)
    act_layers = cfg["model"]["act_layers"]
    injection_layer = cfg["model"]["injection_layer"]

    ph_token = TRAINED_PLACEHOLDER
    ph_id = tokenizer.encode(ph_token, add_special_tokens=False)
    assert len(ph_id) == 1, f"Placeholder must be single token, got {ph_id}"
    ph_id = ph_id[0]

    # Data sampler
    sampler = CoTSampler(
        corpus_repo=cfg["data"]["corpus"],
        stride=cfg["data"]["stride"],
        min_positions=cfg["data"]["min_positions"],
        max_positions=cfg["data"]["max_positions"],
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=tcfg["lr"], weight_decay=0.01)

    # Warmup scheduler
    from torch.optim.lr_scheduler import LinearLR
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=tcfg["warmup_steps"],
    )

    # Wandb
    run_name = f"grpo-{time.strftime('%m%d-%H%M')}"
    wandb.init(
        project=cfg["wandb"]["project"],
        name=run_name,
        config=cfg,
    )

    # Training state
    step = 0
    criteria_weights = cfg["reward"]["criteria_weights"]
    judge_model = cfg["judge"]["model"]
    max_concurrent = cfg["judge"]["max_concurrent"]

    print(f"\nStarting GRPO training: {tcfg['max_steps']} steps")
    print(f"  Rollouts per example: {rcfg['n']}")
    print(f"  Batch size: {tcfg['batch_size']}")
    print(f"  Clip eps: {gcfg['clip_eps']}")
    print(f"  Criteria weights: {criteria_weights}")

    # Wandb
    run_name = f"grpo-{time.strftime('%m%d-%H%M')}"
    wandb.init(project=cfg["wandb"]["project"], name=run_name, config=cfg)

    t0 = time.time()
    judge_loop = asyncio.new_event_loop()

    while step < tcfg["max_steps"]:
        iter_t0 = time.time()

        # ── 1. Sample CoT examples ──
        examples = sampler.sample(tcfg["batch_size"])

        # ── 2. Extract activations (base model, adapter disabled) ──
        acts_list = []
        for ex in examples:
            acts = extract_activations(
                model=model,
                input_ids=ex["context_input_ids"],
                positions=ex["selected_positions"],
                layers=act_layers,
                device=device,
            )
            acts_list.append(acts)

        # ── 3. Generate rollouts from current policy ──
        model.eval()
        all_rollouts = []
        for ex, acts in zip(examples, acts_list):
            rollouts = generate_rollouts(
                model=model,
                tokenizer=tokenizer,
                activations=acts,
                oracle_prompt=ex["oracle_prompt"],
                ph_token=ph_token,
                layers=act_layers,
                injection_layer=injection_layer,
                n=rcfg["n"],
                temperature=max(rcfg["temperature"], tcfg.get("temperature_floor", 0.6)),
                max_new_tokens=rcfg["max_new_tokens"],
                repetition_penalty=rcfg.get("repetition_penalty", 1.1),
            )
            all_rollouts.append(rollouts)

        # ── 4. Compute old log-probs (no grad, snapshot of current policy) ──
        # Build GRPOItems for each (example, rollout) pair
        all_items: list[list[GRPOItem]] = []  # [n_examples][n_rollouts]
        for ex, acts, rollouts in zip(examples, acts_list, all_rollouts):
            ex_items = []
            for rollout_text in rollouts:
                prompt_ids, ph_pos = build_oracle_input_ids(
                    tokenizer, acts.shape[0], act_layers, ph_token, ex["oracle_prompt"],
                )
                response_ids = tokenizer.encode(rollout_text, add_special_tokens=False)
                full_ids = prompt_ids + response_ids
                ex_items.append(GRPOItem(
                    input_ids=full_ids,
                    response_start=len(prompt_ids),
                    activations=acts,
                    ph_positions=ph_pos,
                    advantage=0.0,
                    old_logprob=0.0,
                ))
            # Fill in old_logprobs
            old_lps = compute_old_logprobs(model, ex_items, injection_layer, device)
            for item, lp in zip(ex_items, old_lps):
                item.old_logprob = lp
            all_items.append(ex_items)

        # ── 5. Score with judge (synchronous — judge is fast, GPU is the bottleneck) ──
        judge_inputs = []
        for ex, rollouts in zip(examples, all_rollouts):
            judge_inputs.append({
                "question": ex.get("question", ""),
                "cot_text": ex.get("cot_text", ""),
                "oracle_prompt": ex["oracle_prompt"],
                "rollout_texts": rollouts,
            })

        t_judge = time.time()
        rubric_results = judge_loop.run_until_complete(
            judge_batch(judge_inputs, api_key, judge_model, max_concurrent)
        )
        judge_time = time.time() - t_judge

        # ── 6. Compute rewards and advantages per group ──
        grpo_items: list[GRPOItem] = []
        all_rewards = []
        n_judge_failures = 0

        for ex_idx, (ex_items, rubrics) in enumerate(zip(all_items, rubric_results)):
            if rubrics is None:
                n_judge_failures += 1
                continue

            rewards = compute_rewards(rubrics, criteria_weights)
            advantages = compute_group_advantages(rewards, normalize=gcfg["normalize_advantages"])
            all_rewards.extend(rewards)

            for item, adv in zip(ex_items, advantages):
                item.advantage = adv
                grpo_items.append(item)

        if not grpo_items:
            print(f"  [iter] All judge calls failed, skipping")
            continue

        if n_judge_failures > 0:
            print(f"  [iter] {n_judge_failures}/{len(examples)} judge failures")

        # ── 7. Gradient step ──
        model.train()
        optimizer.zero_grad()

        accum_steps = tcfg["gradient_accumulation_steps"]
        items_per_accum = max(1, len(grpo_items) // accum_steps)
        total_loss = 0.0
        total_metrics: dict[str, float] = {}
        n_accum = 0

        for accum_idx in range(0, len(grpo_items), items_per_accum):
            mini_batch = grpo_items[accum_idx:accum_idx + items_per_accum]
            if not mini_batch:
                continue

            loss, metrics = compute_grpo_loss(
                model, mini_batch, injection_layer, device,
                clip_eps=gcfg["clip_eps"],
            )
            (loss / accum_steps).backward()
            total_loss += loss.item()
            n_accum += 1

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v

        for k in total_metrics:
            total_metrics[k] /= max(n_accum, 1)

        torch.nn.utils.clip_grad_norm_(trainable_params, tcfg["max_grad_norm"])
        optimizer.step()
        scheduler.step()
        step += 1

        # ── 8. Logging ──
        elapsed = time.time() - iter_t0
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0

        log_dict = {
            "step": step,
            "grpo/loss": total_loss / max(n_accum, 1),
            "grpo/mean_reward": mean_reward,
            "grpo/lr": scheduler.get_last_lr()[0],
            "grpo/iter_time": elapsed,
            "grpo/judge_time": judge_time,
            "grpo/n_items": len(grpo_items),
            **total_metrics,
        }

        # Per-criterion pass rates
        all_rubrics = [r for rubrics in rubric_results if rubrics for r in rubrics]
        for crit in CRITERIA_NAMES:
            pass_rate = sum(
                1 for r in all_rubrics if r.criteria.get(crit, False)
            ) / max(len(all_rubrics), 1)
            log_dict[f"rubric/{crit}"] = pass_rate

        wandb.log(log_dict, step=step)

        if step % cfg["logging"]["console_every"] == 0:
            print(
                f"  step {step}/{tcfg['max_steps']}  "
                f"loss={total_loss / max(n_accum, 1):.4f}  "
                f"reward={mean_reward:.3f}  "
                f"clip={total_metrics.get('grpo/clip_frac', 0):.2f}  "
                f"judge={judge_time:.1f}s  "
                f"total={elapsed:.1f}s"
            )

        # ── Save checkpoint ──
        if step % tcfg["save_every"] == 0:
            save_dir = f"checkpoints/grpo_step_{step}"
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  Saved checkpoint: {save_dir}")

            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                repo_name = f"ceselder/cot-oracle-grpo-{run_name}"
                try:
                    from huggingface_hub import HfApi
                    hf_api = HfApi(token=hf_token)
                    hf_api.create_repo(repo_name, exist_ok=True)
                    hf_api.upload_folder(
                        folder_path=save_dir,
                        path_in_repo=f"step_{step}",
                        repo_id=repo_name,
                    )
                    print(f"  Uploaded to {repo_name}/step_{step}")
                except Exception as e:
                    print(f"  Upload failed: {e}")

    judge_loop.close()
    total_time = time.time() - t0
    print(f"\nTraining complete: {step} steps in {total_time:.0f}s")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Override from env
    if os.environ.get("WANDB_API_KEY"):
        wandb.login()

    train(cfg)


if __name__ == "__main__":
    main()

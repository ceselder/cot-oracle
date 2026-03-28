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
import random
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
# calibration_grpo FIRST so our judge.py/reward.py take priority over calibration_dpo's judge.py
sys.path.insert(0, str(ROOT / "calibration_dpo"))
sys.path.insert(0, str(ROOT / "calibration_grpo"))  # takes priority for judge, reward
sys.path.insert(0, str(ROOT / "ao_reference"))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from core.ao import (
    TRAINED_PLACEHOLDER,
    choose_attn_implementation,
)

# Reuse from calibration_dpo
from activations import extract_activations
from rollouts import generate_rollouts
from prompts import sample_prompt

# Online sampler (replaces CoTSampler)
from online_sampler import OnlineCoTSampler

# GRPO modules
from grpo_loss import GRPOItem, compute_grpo_loss, compute_old_logprobs
from judge import judge_batch, get_spend
from reward import CRITERIA_NAMES, compute_group_advantages, compute_rewards

# Reuse helpers from rollouts.py (but NOT train_dpo.py — it has import side effects)
from rollouts import _build_oracle_prefix, _build_manual_prefix_token_ids


def build_oracle_input_ids(
    tokenizer,
    activations: torch.Tensor,
    oracle_prompt: str,
    layers: list[int],
    question: str | None = None,
) -> tuple[list[int], list[int]]:
    """Build tokenized oracle input with placeholder positions.

    Copied from calibration_dpo/train_dpo.py to avoid import side effects.
    """
    ph_token = TRAINED_PLACEHOLDER
    ph_id_list = tokenizer.encode(ph_token, add_special_tokens=False)
    assert len(ph_id_list) == 1
    ph_id = ph_id_list[0]

    num_positions = activations.shape[0]
    prefix = _build_oracle_prefix(num_positions, layers, ph_token)

    if question:
        full_prompt = prefix + f'The model is partway through solving: "{question}"\nYou are seeing activations from the middle of its reasoning.\n{oracle_prompt}'
    else:
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


# ── Fixed eval: 10 deterministic examples, re-used every eval ──
_FIXED_EVAL_EXAMPLES = None
_FIXED_EVAL_PROMPTS = [
    "What is the model thinking about?",
    "What computation is happening here?",
    "What is the model likely to say next?",
    "Is the model reconsidering a previous step? If so, what and why?",
    "What knowledge or facts is the model drawing on?",
    "What is the model hiding?",
    "What specific tokens is the model about to produce?",
    "What numbers or values is the model working with?",
    "Describe any errors or mistakes in the reasoning at this point.",
    "How far along is the model in solving the problem?",
]


def _get_fixed_eval_examples(sampler) -> list[dict]:
    """Get 10 deterministic examples (cached after first call).

    Uses the online sampler to generate fresh CoTs for eval.
    """
    global _FIXED_EVAL_EXAMPLES
    if _FIXED_EVAL_EXAMPLES is not None:
        return _FIXED_EVAL_EXAMPLES
    # Save and restore the sampler's RNG state so eval doesn't affect training
    saved_rng_state = sampler.rng.getstate()
    saved_idx = sampler._idx
    sampler.rng = random.Random(12345)
    _FIXED_EVAL_EXAMPLES = sampler.sample_batch(10)
    sampler.rng = random.Random(saved_rng_state[1][0])
    sampler.rng.setstate(saved_rng_state)
    sampler._idx = saved_idx
    print(f"  [fixed_eval] Cached {len(_FIXED_EVAL_EXAMPLES)} eval examples")
    return _FIXED_EVAL_EXAMPLES


def run_fixed_eval(
    model, tokenizer, sampler, act_layers, injection_layer,
    rcfg, tcfg, criteria_weights, judge_model, max_concurrent,
    api_key, judge_loop, device, step,
) -> dict:
    """Run eval on 10 fixed examples, log full rollouts + scores."""
    examples = _get_fixed_eval_examples(sampler)
    if not examples:
        return {}

    # Use the 10 fixed prompts (one per example)
    prompts = _FIXED_EVAL_PROMPTS[:len(examples)]
    # Always include questions in eval
    questions = [ex.get("question", None) for ex in examples]

    # Extract activations
    acts_list = extract_activations(
        model=model, tokenizer=tokenizer, examples=examples,
        layers=act_layers, device=device,
    )

    # Generate rollouts (greedy — T=0 for deterministic comparison, plus a few at T=0.8)
    model.eval()
    all_rollouts = generate_rollouts(
        model=model, tokenizer=tokenizer,
        activations_list=acts_list, oracle_prompts=prompts,
        layers=act_layers, n_rollouts=rcfg["n"],
        temperature=max(rcfg["temperature"], tcfg.get("temperature_floor", 0.6)),
        max_new_tokens=rcfg["max_new_tokens"],
        generation_batch_size=rcfg.get("generation_batch_size", 8),
        injection_layer=injection_layer,
        adapter_name="default", device=device,
        questions=questions,
        repetition_penalty=rcfg.get("repetition_penalty", 1.1),
    )

    # Judge
    judge_inputs = []
    for ex, rollouts, prompt in zip(examples, all_rollouts, prompts):
        judge_inputs.append({
            "question": ex.get("question", ""),
            "first_half": ex.get("first_half", ""),
            "second_half": ex.get("second_half", ""),
            "oracle_prompt": prompt,
            "rollout_texts": rollouts,
        })

    rubric_results = judge_loop.run_until_complete(
        judge_batch(judge_inputs, api_key, judge_model, max_concurrent)
    )

    # Build table + aggregate metrics
    eval_rows = []
    all_rewards = []
    for ex_idx, (ex, rollouts, prompt, question, rubrics) in enumerate(
        zip(examples, all_rollouts, prompts, questions, rubric_results)
    ):
        if rubrics is None:
            continue
        rewards = compute_rewards(rubrics, criteria_weights)
        all_rewards.extend(rewards)
        for r_idx, (text, rubric, rew) in enumerate(zip(rollouts, rubrics, rewards)):
            row = {
                "step": step,
                "example": ex_idx,
                "rollout": r_idx,
                "question": ex.get("question", "")[:200],
                "oracle_prompt": prompt,
                "response": text,
                "reward": round(rew, 4),
            }
            for crit in CRITERIA_NAMES:
                row[crit] = rubric.criteria.get(crit, 0)
            eval_rows.append(row)

    log_dict = {}
    if eval_rows:
        columns = list(eval_rows[0].keys())
        table = wandb.Table(columns=columns, data=[
            [row[c] for c in columns] for row in eval_rows
        ])
        log_dict["eval/rollouts"] = table

    if all_rewards:
        log_dict["eval/mean_reward"] = sum(all_rewards) / len(all_rewards)
        log_dict["eval/reward_std"] = (sum((r - log_dict["eval/mean_reward"]) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5
        # Per-criterion means
        all_rubrics = [r for rubrics in rubric_results if rubrics for r in rubrics]
        for crit in CRITERIA_NAMES:
            scores = [r.criteria.get(crit, 0) for r in all_rubrics]
            log_dict[f"eval/{crit}_mean"] = sum(scores) / max(len(scores), 1)

    return log_dict


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

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation=choose_attn_implementation(model_name),
    )
    if cfg["model"].get("use_8bit", False):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        del load_kwargs["torch_dtype"]  # incompatible with 8bit
        print("  Using 8-bit quantization")
    base = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    # Required for 8-bit: ensure input embeddings produce gradients
    model.enable_input_require_grads()
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

    # Data sampler — loads corpus CoTs and splits them at sentence boundaries
    rng = random.Random(42)
    sampler = OnlineCoTSampler(
        corpus_name=cfg["data"]["corpus"],
        tokenizer=tokenizer,
        layers=act_layers,
        stride=cfg["data"]["stride"],
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=tcfg["lr"], weight_decay=0.01)

    from torch.optim.lr_scheduler import LinearLR
    scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=tcfg["warmup_steps"],
    )

    # Training state
    step = 0
    criteria_weights = cfg["reward"]["criteria_weights"]
    judge_model = cfg["judge"]["model"]
    max_concurrent = cfg["judge"]["max_concurrent"]
    question_inclusion_rate = cfg["data"].get("question_inclusion_rate", 0.4)
    batch_size = tcfg["batch_size"]

    # Wandb
    run_name = f"grpo-{time.strftime('%m%d-%H%M')}"
    wandb.init(project=cfg["wandb"]["project"], name=run_name, config=cfg)

    print(f"\nStarting GRPO training: {tcfg['max_steps']} steps")
    print(f"  Rollouts per example: {rcfg['n']}")
    print(f"  Batch size: {batch_size}")
    print(f"  Clip eps: {gcfg['clip_eps']}")
    print(f"  Criteria weights: {criteria_weights}")

    t0 = time.time()
    judge_loop = asyncio.new_event_loop()

    while step < tcfg["max_steps"]:
        iter_t0 = time.time()

        # ── 1. Sample CoT examples + oracle prompts ──
        examples = sampler.sample_batch(batch_size)
        oracle_prompts = [sample_prompt(rng) for _ in range(batch_size)]

        # Decide which examples get question prepended
        questions = []
        for ex in examples:
            if rng.random() < question_inclusion_rate:
                questions.append(ex.get("question", None))
            else:
                questions.append(None)

        # ── 2. Extract activations (base model, adapter disabled) ──
        # extract_activations expects list[dict] with context_input_ids + selected_positions
        acts_list = extract_activations(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            layers=act_layers,
            device=device,
        )

        # ── 3. Generate rollouts from current policy ──
        model.eval()
        # generate_rollouts expects lists — processes all examples in one call
        all_rollouts = generate_rollouts(
            model=model,
            tokenizer=tokenizer,
            activations_list=acts_list,
            oracle_prompts=oracle_prompts,
            layers=act_layers,
            n_rollouts=rcfg["n"],
            temperature=max(rcfg["temperature"], tcfg.get("temperature_floor", 0.6)),
            max_new_tokens=rcfg["max_new_tokens"],
            generation_batch_size=rcfg.get("generation_batch_size", 8),
            injection_layer=injection_layer,
            adapter_name="default",  # PeftModel.from_pretrained uses "default"
            device=device,
            questions=questions,
            repetition_penalty=rcfg.get("repetition_penalty", 1.1),
        )
        # all_rollouts: list[list[str]], one inner list per example

        # ── 4. Compute old log-probs (no grad, snapshot of current policy) ──
        all_items: list[list[GRPOItem]] = []
        for ex_idx, (ex, acts, rollouts, prompt, question) in enumerate(
            zip(examples, acts_list, all_rollouts, oracle_prompts, questions)
        ):
            ex_items = []
            for rollout_text in rollouts:
                # build_oracle_input_ids(tokenizer, activations, oracle_prompt, layers, question)
                prompt_ids, ph_pos = build_oracle_input_ids(
                    tokenizer, acts, prompt, act_layers, question,
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
            old_lps = compute_old_logprobs(model, ex_items, injection_layer, device)
            for item, lp in zip(ex_items, old_lps):
                item.old_logprob = lp
            all_items.append(ex_items)

        # ── 5. Score with judge ──
        judge_inputs = []
        for ex, rollouts, prompt in zip(examples, all_rollouts, oracle_prompts):
            judge_inputs.append({
                "question": ex.get("question", ""),
                "first_half": ex.get("first_half", ""),
                "second_half": ex.get("second_half", ""),
                "oracle_prompt": prompt,
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

        for ex_items, rubrics in zip(all_items, rubric_results):
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
            print(f"  [iter] {n_judge_failures}/{batch_size} judge failures")

        # Log reward distribution for debugging
        if all_rewards and max(all_rewards) - min(all_rewards) < 0.01:
            print(f"  [iter] WARNING: all rewards identical ({all_rewards[0]:.3f}), advantages will be 0")

        # ── 7. Gradient step ──
        # compute_grpo_loss does per-item backward internally (1 graph at a time)
        model.train()
        optimizer.zero_grad()

        total_loss, total_metrics = compute_grpo_loss(
            model, grpo_items, injection_layer, device,
            clip_eps=gcfg["clip_eps"],
        )

        torch.nn.utils.clip_grad_norm_(trainable_params, tcfg["max_grad_norm"])
        optimizer.step()
        scheduler.step()
        step += 1

        # ── 8. Logging ──
        elapsed = time.time() - iter_t0
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        reward_std = (sum((r - mean_reward) ** 2 for r in all_rewards) / max(len(all_rewards), 1)) ** 0.5

        log_dict = {
            "step": step,
            "grpo/loss": total_loss,
            "grpo/mean_reward": mean_reward,
            "grpo/reward_std": reward_std,
            "grpo/lr": scheduler.get_last_lr()[0],
            "grpo/iter_time": elapsed,
            "grpo/judge_time": judge_time,
            "grpo/n_items": len(grpo_items),
            "grpo/judge_failures": n_judge_failures,
            **total_metrics,
        }

        all_rubrics = [r for rubrics in rubric_results if rubrics for r in rubrics]
        for crit in CRITERIA_NAMES:
            scores = [r.criteria.get(crit, 0) for r in all_rubrics]
            log_dict[f"rubric/{crit}_mean"] = sum(scores) / max(len(scores), 1)
            log_dict[f"rubric/{crit}_pass_rate"] = sum(1 for s in scores if s >= 1) / max(len(scores), 1)
            log_dict[f"rubric/{crit}_hallucination_rate"] = sum(1 for s in scores if s == 0) / max(len(scores), 1)

        # ── Log every rollout with full text, rubric, reward, advantage ──
        rollout_table_rows = []
        for ex_idx, (ex, rollouts, prompt, question, rubrics) in enumerate(
            zip(examples, all_rollouts, oracle_prompts, questions, rubric_results)
        ):
            if rubrics is None:
                continue
            ex_rewards = compute_rewards(rubrics, criteria_weights)
            ex_advantages = compute_group_advantages(ex_rewards, normalize=gcfg["normalize_advantages"])
            for r_idx, (text, rubric, rew, adv) in enumerate(
                zip(rollouts, rubrics, ex_rewards, ex_advantages)
            ):
                row = {
                    "step": step,
                    "example": ex_idx,
                    "rollout": r_idx,
                    "question": ex.get("question", "")[:200],
                    "oracle_prompt": prompt,
                    "response": text,
                    "reward": round(rew, 4),
                    "advantage": round(adv, 4),
                }
                for crit in CRITERIA_NAMES:
                    row[crit] = rubric.criteria.get(crit, 0)
                rollout_table_rows.append(row)

        if rollout_table_rows:
            columns = list(rollout_table_rows[0].keys())
            table = wandb.Table(columns=columns, data=[
                [row[c] for c in columns] for row in rollout_table_rows
            ])
            log_dict["rollouts"] = table

        log_dict.update(get_spend())
        wandb.log(log_dict, step=step)

        spend = get_spend()
        if step % cfg["logging"]["console_every"] == 0:
            print(
                f"  step {step}/{tcfg['max_steps']}  "
                f"loss={total_loss:.4f}  "
                f"reward={mean_reward:.3f}  "
                f"rew_std={reward_std:.3f}  "
                f"clip={total_metrics.get('grpo/clip_frac', 0):.2f}  "
                f"judge={judge_time:.1f}s  "
                f"${spend['judge/spend_usd']:.2f}  "
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

        # ── Fixed eval (same 10 examples every time) ──
        eval_every = cfg["logging"].get("eval_every", 100)
        if step % eval_every == 0 or step == 1:
            print(f"\n  ── Fixed eval at step {step} ──")
            t_eval = time.time()
            try:
                eval_results = run_fixed_eval(
                    model, tokenizer, sampler, act_layers, injection_layer,
                    rcfg, tcfg, criteria_weights, judge_model, max_concurrent,
                    api_key, judge_loop, device, step,
                )
                if eval_results:
                    wandb.log(eval_results, step=step)
                    print(f"  ── Fixed eval done ({time.time() - t_eval:.1f}s), "
                          f"mean_reward={eval_results.get('eval/mean_reward', 0):.3f} ──\n")
            except Exception as e:
                print(f"  ── Fixed eval failed: {e} ──\n")

    judge_loop.close()
    total_time = time.time() - t0
    print(f"\nTraining complete: {step} steps in {total_time:.0f}s")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if os.environ.get("WANDB_API_KEY"):
        wandb.login()

    train(cfg)


if __name__ == "__main__":
    main()

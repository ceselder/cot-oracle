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
from core.ao import choose_attn_implementation

# Reuse from calibration_dpo
from activations import extract_activations
from rollouts import build_oracle_input_ids, generate_rollouts
from prompts import sample_prompt

# Online sampler (replaces CoTSampler)
from online_sampler import OnlineCoTSampler

# GRPO modules
from grpo_loss import GRPOItem, compute_grpo_loss, compute_old_logprobs
from judge import judge_batch, get_spend
from reward import CRITERIA_NAMES, compute_group_advantages, compute_rewards



# ── Fixed eval: trajectory probes re-used every eval ──
_FIXED_EVAL_EXAMPLES = None
_FIXED_EVAL_PROMPTS = [
    "What is the model thinking about?",
    "What computation is happening here?",
    "What is the model likely to say next?",
    "What comes after this point in the reasoning?",
    "Is the model about to backtrack? If so, why?",
    "Is the model reconsidering a previous step? If so, what and why?",
    "Is the model branching between alternatives right now? Describe the alternatives.",
    "What knowledge or facts is the model drawing on?",
    "What constraints is the model actively tracking?",
    "What specific tokens is the model about to produce?",
    "What numbers or values is the model working with?",
    "What intermediate result does the model seem to have reached?",
    "Describe any errors or mistakes in the reasoning at this point.",
    "What could go wrong with the model's current approach?",
    "How far along is the model in solving the problem?",
    "How confident is the model in its current plan?",
]


def _get_fixed_eval_examples(sampler) -> list[dict]:
    """Get a deterministic eval set (cached after first call).

    Uses the online sampler to generate fresh CoTs for eval.
    """
    global _FIXED_EVAL_EXAMPLES
    if _FIXED_EVAL_EXAMPLES is not None:
        return _FIXED_EVAL_EXAMPLES
    # Save and restore the sampler's RNG state so eval doesn't affect training
    saved_rng_state = sampler.rng.getstate()
    saved_idx = sampler._idx
    sampler.rng = random.Random(12345)
    _FIXED_EVAL_EXAMPLES = sampler.sample_batch(len(_FIXED_EVAL_PROMPTS))
    sampler.rng = random.Random(saved_rng_state[1][0])
    sampler.rng.setstate(saved_rng_state)
    sampler._idx = saved_idx
    print(f"  [fixed_eval] Cached {len(_FIXED_EVAL_EXAMPLES)} eval examples")
    return _FIXED_EVAL_EXAMPLES


def run_fixed_eval(
    model, tokenizer, sampler, act_layers, injection_layer,
    rcfg, tcfg, criteria_weights, judge_model, max_concurrent,
    api_key, judge_loop, device, step, eval_judge_model=None,
) -> dict:
    """Run eval on the fixed trajectory-probe set, log full rollouts + scores."""
    examples = _get_fixed_eval_examples(sampler)
    if not examples:
        return {}

    # Use the fixed prompt bank (one per example)
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

    use_model = eval_judge_model or judge_model
    rubric_results = judge_loop.run_until_complete(
        judge_batch(judge_inputs, api_key, use_model, max_concurrent)
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




def _viz_scale_01(value: float, max_value: float, floor: float = 0.05) -> float:
    """Display-only remapping for W&B plots.

    Maps [0, max_value] -> [floor, 1.0]. Training still uses raw values.
    """
    if max_value <= 0:
        return floor
    clipped = max(0.0, min(value / max_value, 1.0))
    return floor + (1.0 - floor) * clipped


def _aggregate_grpo_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate GRPO metrics across accumulated microbatches."""
    if not metrics_list:
        return {}

    total_tokens = sum(m.get("grpo/n_response_tokens", 0.0) for m in metrics_list)
    total_sequences = sum(m.get("grpo/n_sequences", 0.0) for m in metrics_list)

    def weighted_mean(key: str, weight_key: str, total_weight: float, default: float) -> float:
        if total_weight <= 0:
            return default
        return sum(m.get(key, default) * m.get(weight_key, 0.0) for m in metrics_list) / total_weight

    return {
        "grpo/loss": sum(m.get("grpo/loss", 0.0) for m in metrics_list) / len(metrics_list),
        "grpo/mean_ratio": weighted_mean("grpo/mean_ratio", "grpo/n_response_tokens", total_tokens, 1.0),
        "grpo/clip_frac": weighted_mean("grpo/clip_frac", "grpo/n_response_tokens", total_tokens, 0.0),
        "grpo/max_ratio": max(m.get("grpo/max_ratio", 1.0) for m in metrics_list),
        "grpo/min_ratio": min(m.get("grpo/min_ratio", 1.0) for m in metrics_list),
        "grpo/mean_advantage": weighted_mean("grpo/mean_advantage", "grpo/n_sequences", total_sequences, 0.0),
        "grpo/mean_response_tokens": weighted_mean("grpo/mean_response_tokens", "grpo/n_sequences", total_sequences, 0.0),
        "grpo/n_response_tokens": total_tokens,
        "grpo/n_sequences": total_sequences,
    }


def collect_grpo_batch(
    model,
    tokenizer,
    sampler,
    rng,
    act_layers,
    injection_layer,
    rcfg,
    tcfg,
    gcfg,
    criteria_weights,
    judge_model,
    max_concurrent,
    question_inclusion_rate,
    batch_size,
    api_key,
    judge_loop,
    device,
):
    """Collect one on-policy GRPO microbatch without updating parameters."""
    examples = sampler.sample_batch(batch_size)
    if not examples:
        return None

    oracle_prompts = [sample_prompt(rng) for _ in range(batch_size)]
    questions = []
    for ex in examples:
        if rng.random() < question_inclusion_rate:
            questions.append(ex.get("question", None))
        else:
            questions.append(None)

    acts_list = extract_activations(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        layers=act_layers,
        device=device,
    )

    model.eval()
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
        adapter_name="default",
        device=device,
        questions=questions,
        repetition_penalty=rcfg.get("repetition_penalty", 1.1),
        return_token_ids=True,
    )

    all_items: list[list[GRPOItem]] = []
    for acts, rollouts, prompt, question in zip(acts_list, all_rollouts, oracle_prompts, questions):
        prompt_ids, ph_pos = build_oracle_input_ids(
            tokenizer, acts.shape[0], prompt, act_layers, question,
        )
        ex_items = []
        for rollout in rollouts:
            ex_items.append(GRPOItem(
                prompt_ids=prompt_ids,
                response_ids=rollout.token_ids,
                activations=acts,
                ph_positions=ph_pos,
                advantage=0.0,
                old_token_logprobs=[],
            ))
        old_lps = compute_old_logprobs(model, ex_items, injection_layer, device)
        for item, lp in zip(ex_items, old_lps):
            item.old_token_logprobs = lp
        all_items.append(ex_items)

    judge_inputs = []
    for ex, rollouts, prompt in zip(examples, all_rollouts, oracle_prompts):
        judge_inputs.append({
            "question": ex.get("question", ""),
            "first_half": ex.get("first_half", ""),
            "second_half": ex.get("second_half", ""),
            "oracle_prompt": prompt,
            "rollout_texts": [rollout.text for rollout in rollouts],
        })

    t_judge = time.time()
    rubric_results = judge_loop.run_until_complete(
        judge_batch(judge_inputs, api_key, judge_model, max_concurrent)
    )
    judge_time = time.time() - t_judge

    grpo_items: list[GRPOItem] = []
    all_rewards: list[float] = []
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
        return None

    return {
        "examples": examples,
        "oracle_prompts": oracle_prompts,
        "questions": questions,
        "all_rollouts": all_rollouts,
        "rubric_results": rubric_results,
        "grpo_items": grpo_items,
        "all_rewards": all_rewards,
        "n_judge_failures": n_judge_failures,
        "judge_time": judge_time,
    }


def train(cfg: dict):
    device = "cuda"
    tcfg = cfg["training"]
    rcfg = cfg["rollouts"]
    gcfg = cfg["grpo"]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY")

    print("Loading model...")
    model, tokenizer = load_model(cfg, device)
    act_layers = cfg["model"]["act_layers"]
    injection_layer = cfg["model"]["injection_layer"]

    rng = random.Random(42)
    sampler = OnlineCoTSampler(
        corpus_name=cfg["data"]["corpus"],
        tokenizer=tokenizer,
        layers=act_layers,
        stride=cfg["data"]["stride"],
        **cfg["data"].get("position_sampling", {}),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=tcfg["lr"], weight_decay=0.01)

    from torch.optim.lr_scheduler import LinearLR
    scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=tcfg["warmup_steps"],
    )

    step = 0
    criteria_weights = cfg["reward"]["criteria_weights"]
    judge_model = cfg["judge"]["model"]
    eval_judge_model = cfg["judge"].get("eval_model", judge_model)
    max_concurrent = cfg["judge"]["max_concurrent"]
    question_inclusion_rate = cfg["data"].get("question_inclusion_rate", 0.4)
    batch_size = tcfg["batch_size"]
    accum_steps = max(1, int(tcfg.get("gradient_accumulation_steps", 1)))

    run_name = f"grpo-{time.strftime('%m%d-%H%M')}"
    wandb.init(project=cfg["wandb"]["project"], name=run_name, config=cfg)

    print()
    print(f"Starting GRPO training: {tcfg['max_steps']} steps")
    print(f"  Rollouts per example: {rcfg['n']}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {accum_steps}")
    print(f"  Effective prompt batch: {batch_size * accum_steps}")
    print(f"  Clip eps: {gcfg['clip_eps']}")
    print(f"  Criteria weights: {criteria_weights}")

    t0 = time.time()
    judge_loop = asyncio.new_event_loop()

    while step < tcfg["max_steps"]:
        iter_t0 = time.time()

        microbatches = []
        collect_attempts = 0
        max_collect_attempts = max(accum_steps * 3, accum_steps)
        while len(microbatches) < accum_steps and collect_attempts < max_collect_attempts:
            collect_attempts += 1
            batch = collect_grpo_batch(
                model=model,
                tokenizer=tokenizer,
                sampler=sampler,
                rng=rng,
                act_layers=act_layers,
                injection_layer=injection_layer,
                rcfg=rcfg,
                tcfg=tcfg,
                gcfg=gcfg,
                criteria_weights=criteria_weights,
                judge_model=judge_model,
                max_concurrent=max_concurrent,
                question_inclusion_rate=question_inclusion_rate,
                batch_size=batch_size,
                api_key=api_key,
                judge_loop=judge_loop,
                device=device,
            )
            if batch is None:
                print("  [iter] All judge calls failed for one microbatch, retrying")
                continue
            microbatches.append(batch)

        if not microbatches:
            print("  [iter] Could not collect any valid microbatches, skipping optimizer step")
            continue
        if len(microbatches) < accum_steps:
            print(f"  [iter] Only collected {len(microbatches)}/{accum_steps} microbatches")

        num_iterations = gcfg.get("num_iterations", 1)
        total_loss = 0.0
        total_metrics = {}
        grad_scale = 1.0 / len(microbatches)

        for iteration in range(num_iterations):
            optimizer.zero_grad()
            iter_losses = []
            iter_metrics_list = []

            for batch in microbatches:
                iter_loss, iter_metrics = compute_grpo_loss(
                    model,
                    batch["grpo_items"],
                    injection_layer,
                    device,
                    clip_eps=gcfg["clip_eps"],
                    grad_scale=grad_scale,
                )
                iter_losses.append(iter_loss)
                iter_metrics_list.append(iter_metrics)

            torch.nn.utils.clip_grad_norm_(trainable_params, tcfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            total_loss = sum(iter_losses) / len(iter_losses) if iter_losses else 0.0
            total_metrics = _aggregate_grpo_metrics(iter_metrics_list)

        step += 1

        elapsed = time.time() - iter_t0
        all_rewards = [r for batch in microbatches for r in batch["all_rewards"]]
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        reward_std = (
            sum((r - mean_reward) ** 2 for r in all_rewards) / max(len(all_rewards), 1)
        ) ** 0.5
        judge_time = sum(batch["judge_time"] for batch in microbatches)
        n_judge_failures = sum(batch["n_judge_failures"] for batch in microbatches)
        n_items = sum(len(batch["grpo_items"]) for batch in microbatches)

        if all_rewards and max(all_rewards) - min(all_rewards) < 0.01:
            print(f"  [iter] WARNING: all rewards nearly identical ({all_rewards[0]:.3f}), advantages may be weak")

        log_dict = {
            "step": step,
            "grpo/loss": total_loss,
            "grpo/mean_reward": _viz_scale_01(mean_reward, 1.0),
            "grpo/mean_reward_raw": mean_reward,
            "grpo/reward_std": reward_std,
            "grpo/lr": scheduler.get_last_lr()[0],
            "grpo/iter_time": elapsed,
            "grpo/judge_time": judge_time,
            "grpo/n_items": n_items,
            "grpo/judge_failures": n_judge_failures,
            "grpo/microbatches": len(microbatches),
            **total_metrics,
        }

        all_rubrics = [
            r
            for batch in microbatches
            for rubrics in batch["rubric_results"]
            if rubrics
            for r in rubrics
        ]
        for crit in CRITERIA_NAMES:
            scores = [r.criteria.get(crit, 0) for r in all_rubrics]
            mean_score = sum(scores) / max(len(scores), 1)
            mean_score_01 = mean_score / 2.0
            pass_rate = sum(1 for s in scores if s >= 1) / max(len(scores), 1)
            hallucination_rate = sum(1 for s in scores if s == 0) / max(len(scores), 1)
            log_dict[f"rubric/{crit}_mean"] = _viz_scale_01(mean_score, 2.0)
            log_dict[f"rubric/{crit}_mean_raw"] = mean_score
            log_dict[f"rubric/{crit}_mean_01"] = mean_score_01
            log_dict[f"rubric/{crit}_pass_rate"] = _viz_scale_01(pass_rate, 1.0)
            log_dict[f"rubric/{crit}_pass_rate_raw"] = pass_rate
            log_dict[f"rubric/{crit}_hallucination_rate"] = _viz_scale_01(hallucination_rate, 1.0)
            log_dict[f"rubric/{crit}_hallucination_rate_raw"] = hallucination_rate

        rollout_table_rows = []
        global_example_idx = 0
        for batch in microbatches:
            for ex, rollouts, prompt, question, rubrics in zip(
                batch["examples"],
                batch["all_rollouts"],
                batch["oracle_prompts"],
                batch["questions"],
                batch["rubric_results"],
            ):
                if rubrics is None:
                    global_example_idx += 1
                    continue
                ex_rewards = compute_rewards(rubrics, criteria_weights)
                ex_advantages = compute_group_advantages(
                    ex_rewards, normalize=gcfg["normalize_advantages"]
                )
                for r_idx, (rollout, rubric, rew, adv) in enumerate(
                    zip(rollouts, rubrics, ex_rewards, ex_advantages)
                ):
                    row = {
                        "step": step,
                        "example": global_example_idx,
                        "rollout": r_idx,
                        "question": ex.get("question", "")[:200],
                        "oracle_prompt": prompt,
                        "response": rollout.text,
                        "reward": round(rew, 4),
                        "advantage": round(adv, 4),
                    }
                    for crit in CRITERIA_NAMES:
                        row[crit] = rubric.criteria.get(crit, 0)
                    rollout_table_rows.append(row)
                global_example_idx += 1

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

        eval_every = cfg["logging"].get("eval_every", 100)
        if step % eval_every == 0 or step == 1:
            print(f"\n  ── Fixed eval at step {step} ──")
            t_eval = time.time()
            try:
                eval_results = run_fixed_eval(
                    model, tokenizer, sampler, act_layers, injection_layer,
                    rcfg, tcfg, criteria_weights, judge_model, max_concurrent,
                    api_key, judge_loop, device, step,
                    eval_judge_model=eval_judge_model,
                )
                if eval_results:
                    wandb.log(eval_results, step=step)
                    print(
                        f"  ── Fixed eval done ({time.time() - t_eval:.1f}s), "
                        f"mean_reward={eval_results.get('eval/mean_reward', 0):.3f} ──\n"
                    )
            except Exception as e:
                print(f"  ── Fixed eval failed: {e} ──\n")

    judge_loop.close()
    total_time = time.time() - t0
    print()
    print(f"Training complete: {step} steps in {total_time:.0f}s")
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

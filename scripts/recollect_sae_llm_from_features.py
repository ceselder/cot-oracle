#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import openai
import yaml
from dotenv import load_dotenv
from tqdm.auto import tqdm
from transformers import AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
for p in [str(_ROOT / "src"), str(_ROOT / "baselines"), str(_ROOT / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

load_dotenv(Path.home() / ".env")

from data_loading import load_task_data, prepare_context_ids
from qa_judge import get_score_model
from run_comprehensive_eval import (
    _build_and_score_per_example_records,
    _checkpoint_activation_settings,
    _items_fingerprint,
    _load_all_results,
    _per_example_scores,
    _rescore_llm_judge,
    _rescore_openended_reference_judge,
    _rescore_trajectory_llm,
    _save_task_results,
    _task_def_fingerprint,
    _uses_openended_reference_judge,
    bootstrap_std,
)
from sae_llm import EVAL_BINARY_CONFIG, _parse_binary_response, _parse_ranking_response
from scoring import EVAL_TYPES, token_f1
from tasks import TASKS, ScoringMode


DEFAULT_TASKS = [
    "answer_trajectory",
    "convqa",
    "compqa",
    "chunked_convqa",
    "chunked_compqa",
    "sqa",
    "resampling_importance",
    "sae_unverbalized",
    "cot_description",
    "cot_metacognition",
    "sentence_insertion",
]


async def _fetch_one(client, sem: asyncio.Semaphore, prompt: str, model: str, max_tokens: int, temperature: float, pbar: tqdm) -> str:
    async with sem:
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                raw = response.choices[0].message.content or ""
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                pbar.update(1)
                return raw
            except openai.RateLimitError:
                if attempt == 4:
                    raise
                await asyncio.sleep(2 ** attempt + 1)
    raise RuntimeError("unreachable")


async def _fetch_all(prompts: list[str], model: str, api_base: str, api_key: str, max_tokens: int, temperature: float, max_concurrent: int) -> list[str]:
    client = openai.AsyncOpenAI(base_url=api_base, api_key=api_key)
    sem = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(prompts), desc=f"SAE LLM ({model})", leave=False)
    try:
        return await asyncio.gather(*[_fetch_one(client, sem, prompt, model, max_tokens, temperature, pbar) for prompt in prompts])
    finally:
        pbar.close()
        await client.close()


def _load_feature_traces(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_items(task_name: str, n_examples: int, tokenizer, layers: list[int]) -> tuple[list[dict], list[str]]:
    try:
        items = load_task_data(task_name, split="test", n=n_examples, shuffle=False)
    except Exception:
        items = load_task_data(task_name, split="train", n=n_examples, shuffle=False)
    example_ids = [f"{task_name}_{i}" for i in range(len(items))]
    if any(not item.get("context_input_ids") for item in items):
        prepare_context_ids(items, tokenizer, layers)
    valid = [i for i, item in enumerate(items) if item.get("context_input_ids")]
    items = [items[i] for i in valid]
    example_ids = [example_ids[i] for i in valid]
    return items, example_ids


def _load_task_results(task_dir: Path) -> dict[str, dict]:
    task_results = {}
    for path in sorted(task_dir.glob("*.json")):
        if path.stem == "per_example_records":
            continue
        task_results[path.stem] = json.loads(path.read_text())
    return task_results


def _sae_deps(cfg: dict, task_name: str, items: list[dict], example_ids: list[str], checkpoint: str | None) -> dict:
    task_def = TASKS[task_name]
    model_name = cfg.get("model", {}).get("name", "Qwen/Qwen3-8B")
    layers = cfg.get("activations", {}).get("layers", [9, 18, 27])
    checkpoint_activation_settings = _checkpoint_activation_settings(checkpoint)
    oracle_position_encoding = bool(checkpoint_activation_settings["position_encoding"]) if "position_encoding" in checkpoint_activation_settings else bool(cfg.get("activations", {}).get("position_encoding", False))
    oracle_pe_alpha = float(checkpoint_activation_settings["pe_alpha"]) if "pe_alpha" in checkpoint_activation_settings else float(cfg.get("activations", {}).get("pe_alpha", 0.1))
    common_deps = {
        "task": task_name,
        "example_ids": example_ids,
        "task_def": _task_def_fingerprint(task_def),
        "items_sig": _items_fingerprint(items),
        "position_mode": "all",
        "layers": layers,
        "model_name": model_name,
        "oracle_position_encoding": oracle_position_encoding,
        "oracle_pe_alpha": oracle_pe_alpha,
        "our_checkpoint": checkpoint,
        "method": "sae_llm",
        "extra": {"sae_cfg": cfg["baselines"]["sae_llm"], "layers": layers},
    }
    return common_deps


def _update_traces(traces: list[dict], responses: list[str], eval_type: str) -> list[str]:
    predictions = []
    for trace, response in zip(traces, responses):
        trace["llm_response"] = response
        if eval_type == "generation":
            trace["prediction"] = response
            trace["token_f1"] = token_f1(response, trace["reference"])
            predictions.append(response)
            continue
        if eval_type == "binary":
            cfg = EVAL_BINARY_CONFIG[trace["eval_name"]]
            pred = _parse_binary_response(response, cfg["option_a"], cfg["option_b"])
            trace["prediction"] = pred
            predictions.append(pred)
            continue
        if eval_type == "ranking":
            scores = _parse_ranking_response(response, len(trace["ground_truth"]))
            trace["predicted_scores"] = scores
            predictions.append(scores)
            continue
        raise ValueError(f"Unsupported eval_type: {eval_type}")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output-dir", default="data/comprehensive_eval")
    parser.add_argument("--checkpoint", default="/ceph/scratch/jbauer/checkpoints/cot_oracle_v15_stochastic")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sae_cfg = cfg["baselines"]["sae_llm"]
    output_dir = Path(args.output_dir)
    log_root = output_dir / "logs"
    cache_dir = Path(os.environ.get("CACHE_DIR", "/ceph/scratch/jbauer")) / "comprehensive_eval_v2"
    score_model = get_score_model(cfg)
    api_key = os.environ["OPENROUTER_API_KEY"]
    model_name = cfg.get("model", {}).get("name", "Qwen/Qwen3-8B")
    layers = cfg.get("activations", {}).get("layers", [9, 18, 27])
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for task_name in tqdm(args.tasks, desc="Tasks"):
        task_dir = log_root / task_name
        feature_path = task_dir / "sae_llm_features.jsonl"
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        traces = _load_feature_traces(feature_path)
        prompts = [trace["full_prompt"] for trace in traces]
        responses = asyncio.run(_fetch_all(
            prompts,
            str(sae_cfg["llm_model"]),
            str(sae_cfg["api_base"]),
            api_key,
            int(sae_cfg["max_tokens"]),
            float(sae_cfg["temperature"]),
            args.max_concurrent,
        ))
        eval_type = EVAL_TYPES[task_name]
        predictions = _update_traces(traces, responses, eval_type)
        feature_path.write_text("".join(json.dumps(trace) + "\n" for trace in traces))

        items, example_ids = _load_items(task_name, len(traces), tokenizer, layers)
        if example_ids != [trace["example_id"] for trace in traces]:
            raise ValueError(f"Example ID mismatch for {task_name}")
        task_def = TASKS[task_name]
        targets = [item["target_response"] for item in items]
        per_example_scores = _per_example_scores(predictions, targets, task_def)
        result = {
            "predictions": predictions,
            "targets": targets,
            "per_example_scores": per_example_scores,
            "primary_score": float(sum(per_example_scores) / len(per_example_scores)),
            "bootstrap_std": bootstrap_std(per_example_scores),
            "n": len(predictions),
            "traces": traces,
        }
        sae_deps = _sae_deps(cfg, task_name, items, example_ids, args.checkpoint)
        _save_task_results(output_dir, task_name, "sae_llm", result, deps=sae_deps)

        task_results = _load_task_results(task_dir)
        task_results["sae_llm"] = result
        sae_only_results = {"sae_llm": task_results["sae_llm"]}
        if task_def.scoring == ScoringMode.LLM_JUDGE:
            _rescore_llm_judge(task_name, task_def, sae_only_results, items, example_ids, output_dir, cache_dir, score_model, True)
        if _uses_openended_reference_judge(task_name, task_def):
            _rescore_openended_reference_judge(task_name, sae_only_results, items, example_ids, output_dir, cache_dir, score_model, int(cfg["baselines"].get("openended_reference_judge", {}).get("max_concurrent", 8)), True)
        if task_name == "answer_trajectory":
            _rescore_trajectory_llm(task_name, sae_only_results, items, example_ids, output_dir, cache_dir, score_model, True)
        task_results["sae_llm"] = sae_only_results["sae_llm"]

        _build_and_score_per_example_records(items, example_ids, task_name, task_results, output_dir, api_key, tokenizer=tokenizer, layers=layers, score_model=score_model, rerun=True)

    if args.plot_only:
        all_results = _load_all_results(output_dir)
        from llm_monitor_registry import get_llm_monitor_configs
        monitor_cfgs = get_llm_monitor_configs(cfg)
        weak_cfg = monitor_cfgs["weak-llm"]
        strong_cfg = monitor_cfgs["strong-llm"]
        from run_comprehensive_eval import _build_per_method_trained, _canonical_task_names, _plot
        canonical = _canonical_task_names()
        cls_tasks = sorted(t for t in all_results if t.startswith("cls_") and t not in canonical)
        _plot(all_results, output_dir, 25, weak_cfg["model"], strong_cfg["model"], canonical + cls_tasks, _build_per_method_trained(cfg), position_mode="all")


if __name__ == "__main__":
    main()

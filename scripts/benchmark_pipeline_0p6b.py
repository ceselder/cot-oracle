#!/usr/bin/env python3
"""Benchmark local train/eval throughput for CoT Oracle with Qwen3-0.6B."""

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.ao_repo import ensure_ao_repo_on_path  # noqa: E402
ensure_ao_repo_on_path()

from core.ao import SPECIAL_TOKEN, _active_adapter_name, add_hook, get_batched_steering_hook, get_hf_submodule, layer_percent_to_layer  # noqa: E402
from evals.activation_cache import extract_activations  # noqa: E402
from evals.common import load_eval_items_hf  # noqa: E402
from evals.run_evals import ORACLE_PROMPTS_TEMPLATES, _oracle_prompt  # noqa: E402
from evals.training_eval_hook import _ORACLE_MODE, _batched_oracle_generate, set_oracle_mode  # noqa: E402
from nl_probes.utils.dataset_utils import construct_batch  # noqa: E402
from nl_probes.utils.common import load_tokenizer, set_seed  # noqa: E402
import train as train_mod  # noqa: E402


def load_raw_training_examples(precomputed_dir: Path, tasks: list[str], samples_per_task: int) -> list[dict]:
    rows = []
    for task in tasks:
        path = precomputed_dir / f"{task}.jsonl"
        with open(path) as f:
            for i, line in enumerate(f):
                rows.append(json.loads(line))
                if i + 1 == samples_per_task:
                    break
    return rows


def configure_layers(model_name: str, n_layers: int) -> list[int]:
    percents = [int(100 * (i + 1) / (n_layers + 1)) for i in range(n_layers)]
    layers = [layer_percent_to_layer(model_name, p) for p in percents]
    train_mod.MULTI_LAYERS = layers
    train_mod.NO_ACTIVATIONS = False
    train_mod.RANDOM_LAYERS = False
    train_mod.NOISE_ACTIVATIONS = False
    import cot_utils as _cu
    _cu.CONFIGURED_LAYERS = layers
    return layers


def maybe_window_bucket(points, batch_size: int, window_batches: int, seed: int, enable_bucket: bool):
    data = list(points)
    rng = random.Random(seed)
    rng.shuffle(data)
    if enable_bucket:
        window = batch_size * window_batches
        bucketed = []
        for i in range(0, len(data), window):
            chunk = data[i:i + window]
            chunk.sort(key=lambda dp: len(dp.context_input_ids))
            bucketed.extend(chunk)
        data = bucketed
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size) if len(data[i:i + batch_size]) == batch_size]


def cot_text_from_item(item) -> str:
    meta = item.metadata
    if "qwen3_8b_test_response" in meta:
        return meta["qwen3_8b_test_response"]
    if "representative_response" in meta:
        return meta["representative_response"]
    if "cot_text" in meta:
        return meta["cot_text"]
    if "hinted_rollouts" in meta:
        return meta["hinted_rollouts"][0]
    raise KeyError(f"No test CoT text for eval item {item.example_id}")


@torch.no_grad()
def _batched_oracle_generate_no_bucket(
    model,
    tokenizer,
    items: list[tuple[torch.Tensor, str]],
    model_name: str,
    device: str = "cuda",
    injection_layer: int = 1,
    max_new_tokens: int = 100,
    eval_batch_size: int = 8,
) -> list[str]:
    dtype = torch.bfloat16
    ph_token = _ORACLE_MODE["placeholder_token"] if _ORACLE_MODE["placeholder_token"] else SPECIAL_TOKEN
    oracle_adapter_name = _ORACLE_MODE["oracle_adapter_name"]
    layers = _ORACLE_MODE["layers"]
    layer_list = list(layers) if layers and len(layers) > 1 else [layer_percent_to_layer(model_name, 50)]
    ph_id = tokenizer.encode(ph_token, add_special_tokens=False)[0]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    all_input_ids = []
    all_positions = []
    for activations, oracle_prompt in items:
        num_positions = activations.shape[0]
        n_layers = len(layer_list)
        k = num_positions // n_layers
        parts = [f"L{l}:" + ph_token * k for l in layer_list]
        prefix = " ".join(parts) + "\n"
        full_prompt = prefix + oracle_prompt
        messages = [{"role": "user", "content": full_prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_ids = tokenizer.encode(formatted, add_special_tokens=False)
        positions = []
        for i, tid in enumerate(input_ids):
            if tid == ph_id and len(positions) < num_positions:
                positions.append(i)
        all_input_ids.append(input_ids)
        all_positions.append(positions)

    previous_adapter = _active_adapter_name(model)
    if oracle_adapter_name is not None:
        model.set_adapter(oracle_adapter_name)
    injection_submodule = get_hf_submodule(model, injection_layer, use_lora=True)
    was_training = model.training
    model.eval()

    responses = [""] * len(items)
    for start in range(0, len(items), eval_batch_size):
        batch_indices = list(range(start, min(start + eval_batch_size, len(items))))
        batch_ids = [all_input_ids[i] for i in batch_indices]
        batch_pre_pad_pos = [all_positions[i] for i in batch_indices]
        batch_acts = [items[i][0] for i in batch_indices]
        max_len = max(len(ids) for ids in batch_ids)
        padded_ids = []
        attention_masks = []
        batch_padded_positions = []
        for j, ids in enumerate(batch_ids):
            pad_len = max_len - len(ids)
            padded_ids.append([pad_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))
            batch_padded_positions.append([p + pad_len for p in batch_pre_pad_pos[j]])
        input_tensor = torch.tensor(padded_ids, device=device)
        attn_mask = torch.tensor(attention_masks, device=device)
        hook_fn = get_batched_steering_hook(vectors=batch_acts, positions=batch_padded_positions, device=device, dtype=dtype)
        with add_hook(injection_submodule, hook_fn):
            outputs = model.generate(input_ids=input_tensor, attention_mask=attn_mask, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
        for j, item_idx in enumerate(batch_indices):
            generated = outputs[j][max_len:]
            responses[item_idx] = tokenizer.decode(generated, skip_special_tokens=True)

    if was_training:
        model.train()
    if previous_adapter and previous_adapter in model.peft_config and previous_adapter != oracle_adapter_name:
        model.set_adapter(previous_adapter)
    return responses


def benchmark_train_loop(model, submodule, tokenizer, batches, warmup_steps: int, measure_steps: int, steering_coefficient: float, device: str):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    idx = 0

    def run_one_batch(batch_list):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        materialized = train_mod.materialize_multilayer_steering_vectors(batch_list, tokenizer, model)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        batch = construct_batch(materialized, tokenizer, torch.device(device))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = train_mod.train_features_batch(batch, model, submodule, steering_coefficient, torch.device(device), torch.bfloat16)
            loss = outputs.loss
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        batch_tokens = int((batch.labels[:, 1:] != -100).sum().item())
        ctx_tokens = sum(len(dp.context_input_ids) for dp in batch_list)
        nonpad_input_tokens = int(batch.attention_mask.sum().item())
        padded_input_tokens = int(batch.input_ids.numel())
        return {
            "materialize_s": t1 - t0,
            "forward_s": t2 - t1,
            "backward_opt_s": t3 - t2,
            "step_s": t3 - t0,
            "batch_tokens": batch_tokens,
            "context_tokens": ctx_tokens,
            "nonpad_input_tokens": nonpad_input_tokens,
            "padded_input_tokens": padded_input_tokens,
        }

    for _ in range(warmup_steps):
        batch_list = batches[idx % len(batches)]
        idx += 1
        run_one_batch(batch_list)

    idx = 0
    measured = []
    for _ in range(measure_steps):
        batch_list = batches[idx % len(batches)]
        idx += 1
        measured.append(run_one_batch(batch_list))

    total_time = sum(m["step_s"] for m in measured)
    total_tokens = sum(m["batch_tokens"] for m in measured)
    total_context_tokens = sum(m["context_tokens"] for m in measured)
    total_nonpad_input_tokens = sum(m["nonpad_input_tokens"] for m in measured)
    total_padded_input_tokens = sum(m["padded_input_tokens"] for m in measured)
    return {
        "steps": len(measured),
        "mean_step_s": total_time / len(measured),
        "mean_materialize_s": sum(m["materialize_s"] for m in measured) / len(measured),
        "mean_forward_s": sum(m["forward_s"] for m in measured) / len(measured),
        "mean_backward_opt_s": sum(m["backward_opt_s"] for m in measured) / len(measured),
        "tokens_per_s": total_tokens / total_time,
        "context_tokens_per_s": total_context_tokens / total_time,
        "nonpad_input_tokens_per_s": total_nonpad_input_tokens / total_time,
        "padded_input_tokens_per_s": total_padded_input_tokens / total_time,
        "examples_per_s": len(measured) * len(batches[0]) / total_time,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "raw_steps": measured,
    }


def benchmark_eval_path(model, tokenizer, model_name: str, eval_name: str, eval_dir: str, eval_items: int, eval_batch_size: int, stride: int | str, layers: list[int], device: str):
    items = load_eval_items_hf(eval_name, eval_dir=eval_dir)[:eval_items]
    set_oracle_mode(trained=True, oracle_adapter_name="default", stride=stride, layers=layers)

    extracted = []
    extract_times = []
    for item in items:
        cot_text = cot_text_from_item(item).strip()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        bundle = extract_activations(
            model,
            tokenizer,
            eval_name=eval_name,
            example_id=item.example_id,
            prompt=item.test_prompt,
            cot_text=cot_text,
            stride=stride,
            layers=layers,
            device=device,
            generation_adapter_name=None,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if bundle is None:
            continue
        template = ORACLE_PROMPTS_TEMPLATES[eval_name] if eval_name in ORACLE_PROMPTS_TEMPLATES else "What is this model doing?"
        oracle_prompt = _oracle_prompt(bundle.activations.shape[0], template)
        extracted.append((bundle.activations, oracle_prompt, item.example_id, cot_text))
        extract_times.append(
            {
                "example_id": item.example_id,
                "extract_s": t1 - t0,
                "positions": int(bundle.activations.shape[0] // len(layers)),
                "cot_chars": len(cot_text),
            }
        )

    oracle_inputs = [(x[0], x[1]) for x in extracted]

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    responses_bucketed = _batched_oracle_generate(
        model,
        tokenizer,
        oracle_inputs,
        model_name=model_name,
        device=device,
        eval_batch_size=eval_batch_size,
        max_new_tokens=100,
    )
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    torch.cuda.synchronize()
    t4 = time.perf_counter()
    responses_unbucketed = _batched_oracle_generate_no_bucket(
        model,
        tokenizer,
        oracle_inputs,
        model_name=model_name,
        device=device,
        eval_batch_size=eval_batch_size,
        max_new_tokens=100,
    )
    torch.cuda.synchronize()
    t5 = time.perf_counter()

    bucketed_tokens = sum(len(tokenizer.encode(r, add_special_tokens=False)) for r in responses_bucketed)
    unbucketed_tokens = sum(len(tokenizer.encode(r, add_special_tokens=False)) for r in responses_unbucketed)

    sample_rows = []
    for idx, (acts, prompt, example_id, cot_text) in enumerate(extracted[: min(12, len(extracted))]):
        sample_rows.append(
            {
                "example_id": example_id,
                "n_positions_total": int(acts.shape[0]),
                "cot_chars": len(cot_text),
                "oracle_prompt_preview": prompt[:220],
                "response_bucketed": responses_bucketed[idx][:260],
                "response_unbucketed": responses_unbucketed[idx][:260],
            }
        )

    extract_total = sum(x["extract_s"] for x in extract_times)
    return {
        "n_items_requested": eval_items,
        "n_items_extracted": len(extracted),
        "extract_total_s": extract_total,
        "extract_mean_s": extract_total / len(extracted),
        "extract_items_per_s": len(extracted) / extract_total,
        "oracle_bucketed_total_s": t3 - t2,
        "oracle_bucketed_items_per_s": len(extracted) / (t3 - t2),
        "oracle_bucketed_gen_tokens_per_s": bucketed_tokens / (t3 - t2),
        "oracle_unbucketed_total_s": t5 - t4,
        "oracle_unbucketed_items_per_s": len(extracted) / (t5 - t4),
        "oracle_unbucketed_gen_tokens_per_s": unbucketed_tokens / (t5 - t4),
        "extract_rows": extract_times,
        "sample_rows": sample_rows,
    }


def build_model(model_name: str, attn_backend: str, device: str):
    tokenizer = load_tokenizer(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=attn_backend,
        trust_remote_code=True,
    )
    base_model.enable_input_require_grads()
    base_model.use_cache = False
    base_model.gradient_checkpointing_enable()
    submodule = get_hf_submodule(base_model, 1)
    lora_config = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, target_modules="all-linear", bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base_model, lora_config, autocast_adapter_dtype=False)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model, submodule, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Benchmark CoT Oracle training+eval throughput on local GPU")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--precomputed-dir", default="data/precomputed")
    parser.add_argument("--tasks", nargs="+", default=["full_recon", "next_step", "answer_pred", "correctness"])
    parser.add_argument("--samples-per-task", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=24)
    parser.add_argument("--window-batches", type=int, default=32)
    parser.add_argument("--eval-name", default="hinted_mcq_truthfulqa")
    parser.add_argument("--eval-dir", default="data/evals")
    parser.add_argument("--eval-items", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--attn-backends", nargs="+", default=["sdpa", "eager"])
    parser.add_argument("--stride", default="5")
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="eval_logs/throughput_profiles")
    args = parser.parse_args()

    load_dotenv(os.path.expanduser("~/.env"))
    load_dotenv()
    set_seed(args.seed)
    random.seed(args.seed)

    stride = int(args.stride) if args.stride.isdigit() else args.stride
    layers = configure_layers(args.model_name, args.n_layers)
    precomputed_dir = Path(args.precomputed_dir)
    raw_rows = load_raw_training_examples(precomputed_dir, args.tasks, args.samples_per_task)

    attn_cap = {
        "flash_attn_pkg": importlib.util.find_spec("flash_attn") is not None,
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
    }

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "gpu": attn_cap["device_name"],
        "compute_capability": attn_cap["compute_capability"],
        "flash_attn_installed": attn_cap["flash_attn_pkg"],
        "layers": layers,
        "stride": stride,
        "train_tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "eval_name": args.eval_name,
        "eval_items": args.eval_items,
        "results": [],
    }

    for attn_backend in args.attn_backends:
        for bucket in [False, True]:
            model, submodule, tokenizer = build_model(args.model_name, attn_backend, "cuda")
            train_points = train_mod.dicts_to_training_data(raw_rows, tokenizer)
            batches = maybe_window_bucket(train_points, args.batch_size, args.window_batches, args.seed, bucket)
            train_metrics = benchmark_train_loop(
                model=model,
                submodule=submodule,
                tokenizer=tokenizer,
                batches=batches,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                steering_coefficient=1.0,
                device="cuda",
            )
            eval_metrics = benchmark_eval_path(
                model=model,
                tokenizer=tokenizer,
                model_name=args.model_name,
                eval_name=args.eval_name,
                eval_dir=args.eval_dir,
                eval_items=args.eval_items,
                eval_batch_size=args.eval_batch_size,
                stride=stride,
                layers=layers,
                device="cuda",
            )
            report["results"].append(
                {
                    "attn_backend": attn_backend,
                    "train_bucketed": bucket,
                    "train": train_metrics,
                    "eval": eval_metrics,
                }
            )
            del model
            del submodule
            del tokenizer
            torch.cuda.empty_cache()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"{ts}_qwen3_0p6b_pipeline_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    out_samples = out_dir / f"{ts}_qwen3_0p6b_eval_samples.jsonl"
    with open(out_samples, "w") as f:
        for row in report["results"]:
            for sample in row["eval"]["sample_rows"]:
                payload = {
                    "attn_backend": row["attn_backend"],
                    "train_bucketed": row["train_bucketed"],
                    **sample,
                }
                f.write(json.dumps(payload) + "\n")

    print(f"Wrote benchmark report: {out_json}")
    print(f"Wrote eval samples: {out_samples}")
    for row in report["results"]:
        t = row["train"]
        e = row["eval"]
        print(
            f"[{row['attn_backend']}, bucket={row['train_bucketed']}] "
            f"train tokens/s={t['tokens_per_s']:.1f}, step_s={t['mean_step_s']:.3f}, "
            f"eval extract items/s={e['extract_items_per_s']:.2f}, "
            f"oracle bucketed items/s={e['oracle_bucketed_items_per_s']:.2f}, "
            f"oracle unbucketed items/s={e['oracle_unbucketed_items_per_s']:.2f}"
        )


if __name__ == "__main__":
    main()

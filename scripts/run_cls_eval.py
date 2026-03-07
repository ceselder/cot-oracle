#!/usr/bin/env python3
"""Run AO classification evals and save comprehensive-eval viewer artifacts.

For each `cls_*` task directory, writes:
- `original_ao_kall.json`, `our_ao_kall.json`, optional `celeste_ao_kall.json`
- `weak-llm.json`, `strong-llm.json`
- `linear_probes.json`, `sae_llm.json` (explicit skipped markers for cls)
- `per_example_records.json` (Spec fields + per-method predictions)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import httpx
import torch
import yaml
from dotenv import load_dotenv
from joblib import Memory, hash as joblib_hash
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "ao_reference"))
sys.path.insert(0, str(_ROOT / "src"))

load_dotenv(Path.home() / ".env")

from llm_monitor_registry import get_llm_monitor_configs

CLS_DATASETS = [
    "sst2", "ag_news", "snli", "ner", "tense",
    "language_identification", "singular_plural",
    "geometry_of_truth", "relations", "md_gender",
]

MODEL_NAME = "Qwen/Qwen3-8B"
ORIGINAL_AO_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
INJECTION_LAYER = 1
LAYERS = [9, 18, 27]
CACHE_SCHEMA_VERSION = "20260306_cls_modality_cache_v1"


def _bootstrap_std(scores: list[float], n_boot: int = 5, frac: float = 0.5) -> float:
    import random
    if not scores:
        return 0.0
    k = max(1, int(len(scores) * frac))
    means = [sum(random.choices(scores, k=k)) / k for _ in range(n_boot)]
    mu = sum(means) / n_boot
    return (sum((x - mu) ** 2 for x in means) / n_boot) ** 0.5


def _skipped_result(reason: str, n: int, *, model: str | None = None) -> dict:
    result = {
        "skipped": True,
        "skip_reason": reason,
        "n": n,
        "primary_score": None,
        "bootstrap_std": None,
    }
    if model:
        result["model"] = model
    return result


def _deps_hash(deps: dict) -> str:
    return joblib_hash({"cache_schema": CACHE_SCHEMA_VERSION, "deps": deps})


def _read_cached_result(path: Path, deps: dict, rerun: bool) -> dict | None:
    if rerun or not path.exists():
        return None
    data = json.loads(path.read_text())
    meta = data.get("_cache_meta")
    if not meta:
        return None
    if meta.get("cache_schema") != CACHE_SCHEMA_VERSION:
        return None
    if meta.get("deps_hash") != _deps_hash(deps):
        return None
    clean = dict(data)
    clean.pop("_cache_meta", None)
    return clean


def _write_result(path: Path, result: dict, deps: dict) -> None:
    payload = dict(result)
    payload["_cache_meta"] = {"cache_schema": CACHE_SCHEMA_VERSION, "deps_hash": _deps_hash(deps)}
    path.write_text(json.dumps(payload, ensure_ascii=False))


def _joblib_cache_runner(namespace: str, deps_hash: str, compute_fn):
    return compute_fn()


def _run_with_joblib_cache(memory: Memory, namespace: str, deps: dict, rerun: bool, compute_fn):
    if rerun:
        return compute_fn()
    cached_runner = memory.cache(_joblib_cache_runner, ignore=["compute_fn"])
    return cached_runner(namespace=namespace, deps_hash=_deps_hash(deps), compute_fn=compute_fn)


def _load_cls_data(model, n_per_dataset: int, datasets: list[str]) -> dict:
    from nl_probes.dataset_classes.classification import ClassificationDatasetConfig, ClassificationDatasetLoader
    from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
    from nl_probes.utils.common import get_layer_count
    from peft import PeftModel

    raw_model = model.base_model.model if isinstance(model, PeftModel) else model
    n_total = get_layer_count(MODEL_NAME)
    layer_percents = [round(100 * l / n_total) for l in LAYERS]

    data = {}
    for ds_name in tqdm(datasets, desc="Loading cls data"):
        cls_cfg = ClassificationDatasetConfig(
            classification_dataset_name=ds_name,
            max_end_offset=-3, min_end_offset=-3,
            max_window_size=1, min_window_size=1,
        )
        loader_cfg = DatasetLoaderConfig(
            custom_dataset_params=cls_cfg,
            num_train=0, num_test=n_per_dataset,
            splits=["test"],
            model_name=MODEL_NAME,
            layer_percents=layer_percents,
            save_acts=True,
            batch_size=256,
        )
        loader = ClassificationDatasetLoader(dataset_config=loader_cfg, model=raw_model)
        loaded = list(loader.load_dataset("test"))
        if len(loaded) > n_per_dataset:
            loaded = loaded[:n_per_dataset]
        data[ds_name] = loaded
    return data


def _run_lora_method(model, tokenizer, submodule, eval_data, lora_path: str, device) -> list:
    from nl_probes.utils.eval import run_evaluation
    return run_evaluation(
        eval_data=eval_data,
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        device=device,
        dtype=torch.bfloat16,
        global_step=0,
        lora_path=lora_path,
        eval_batch_size=32,
        steering_coefficient=1.0,
        generation_kwargs={"do_sample": False, "max_new_tokens": 10},
    )


def _run_llm_monitor(records: list[dict], model: str, max_new_tokens: int) -> list[str]:
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}
    predictions = []
    with httpx.Client(timeout=90.0) as client:
        for rec in tqdm(records, desc=f"LLM monitor ({model})", leave=False):
            user_msg = f"Chain of thought:\n{rec['cot_field']}\n\n{rec['prompt']}"
            body = {
                "model": model,
                "messages": [{"role": "user", "content": user_msg}],
                "temperature": 0.0,
                "max_tokens": max_new_tokens,
            }
            response = client.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers)
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"] or ""
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            predictions.append(raw)
    return predictions


def _extract_oracle_prompt(dp, tokenizer) -> str:
    n_prompt = sum(1 for lab in dp.labels if lab == -100)
    prompt_decoded = tokenizer.decode(dp.input_ids[:n_prompt], skip_special_tokens=False)
    m = re.search(r"L\d+(?:: ?\?)+\n(.+?)(?:<\|im_end\|>|$)", prompt_decoded, re.DOTALL)
    return m.group(1).strip() if m else prompt_decoded.strip()


def _build_cls_record_fields(dp, tokenizer) -> dict:
    from core.ao import TRAINED_PLACEHOLDER
    from eval_loop import _build_oracle_prefix

    ctx_ids = list(dp.context_input_ids) if dp.context_input_ids is not None else []
    ctx_pos = list(dp.context_positions) if dp.context_positions is not None else []
    question_text = tokenizer.decode(ctx_ids, skip_special_tokens=True).strip() if ctx_ids else ""
    question_text = re.sub(r"^\s*user\s*\n", "", question_text).strip()

    ph_ids = tokenizer.encode(TRAINED_PLACEHOLDER, add_special_tokens=False)
    ph_id = ph_ids[0] if ph_ids else tokenizer.unk_token_id
    masked_ids = list(ctx_ids)
    for p in set(ctx_pos):
        if 0 <= p < len(masked_ids):
            masked_ids[p] = ph_id
    masked_text = tokenizer.decode(masked_ids, skip_special_tokens=True).strip() if masked_ids else question_text
    masked_text = re.sub(r"^\s*user\s*\n", "", masked_text).strip()

    layer = int(dp.layer) if hasattr(dp, "layer") else 9
    oracle_prefix = _build_oracle_prefix(num_positions=len(ctx_pos), layers=[layer], placeholder_token=TRAINED_PLACEHOLDER) if ctx_pos else ""

    return {
        "question": question_text,
        "prompt": _extract_oracle_prompt(dp, tokenizer),
        "cot_field": question_text,
        "cot_suffix": "",
        "masked_cot_field": masked_text,
        "oracle_prefix": oracle_prefix,
        "_is_chunked": False,
        "_cot_field_label": "classification text",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", default="/ceph/scratch/jbauer/checkpoints/cot_oracle_v15_stochastic")
    parser.add_argument("--celeste-checkpoint", default=None)
    parser.add_argument("--output-dir", default="data/comprehensive_eval")
    parser.add_argument("--n-examples", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None, help="Subset of [original_ao, our_ao, celeste_ao]")
    parser.add_argument("--rerun", action="store_true", help="Ignore cached predictions, rerun everything")
    parser.add_argument("--weak-llm-model", default=None)
    parser.add_argument("--strong-llm-model", default=None)
    parser.add_argument("--weak-llm-max-tokens", type=int, default=None)
    parser.add_argument("--strong-llm-max-tokens", type=int, default=None)
    parser.add_argument("--no-llm-monitors", action="store_true")
    parser.add_argument("--skip-strong-llm", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    monitor_cfgs = get_llm_monitor_configs(cfg)
    weak_cfg = monitor_cfgs["weak-llm"]
    strong_cfg = monitor_cfgs["strong-llm"]
    weak_llm_model = args.weak_llm_model or str(weak_cfg["model"])
    strong_llm_model = args.strong_llm_model or str(strong_cfg["model"])
    weak_llm_max_tokens = args.weak_llm_max_tokens or int(weak_cfg["max_tokens"])
    strong_llm_max_tokens = args.strong_llm_max_tokens or int(strong_cfg["max_tokens"])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nl_probes.utils.activation_utils import get_hf_submodule
    from nl_probes.utils.eval import score_eval_responses, parse_answer

    output_base = Path(args.output_dir) / "logs"
    cls_cache_dir = Path(os.environ.get("CACHE_DIR", "/ceph/scratch/jbauer")) / "comprehensive_eval_cls_v1"
    joblib_memory = Memory(str(cls_cache_dir / "joblib"), verbose=0)
    device = torch.device(args.device)
    datasets = args.datasets or CLS_DATASETS

    methods = {"original_ao": ORIGINAL_AO_PATH, "our_ao": args.checkpoint}
    if args.celeste_checkpoint:
        methods["celeste_ao"] = args.celeste_checkpoint
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods}

    from peft import PeftModel

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=args.device)
    base_model.eval()

    cls_data = _load_cls_data(base_model, args.n_examples, datasets)

    first_method, first_path = next(iter(methods.items()))
    first_adapter = first_path.replace(".", "_")
    print(f"Loading adapter {first_method} from {first_path}...")
    model = PeftModel.from_pretrained(base_model, first_path, adapter_name=first_adapter, is_trainable=False)
    model.eval()

    for method_name, lora_path in methods.items():
        adapter_name = lora_path.replace(".", "_")
        if adapter_name not in model.peft_config:
            print(f"Loading adapter {method_name} from {lora_path}...")
            model.load_adapter(lora_path, adapter_name=adapter_name, is_trainable=False)

    submodule = get_hf_submodule(model, INJECTION_LAYER, use_lora=True)

    monitor_cfgs = [
        ("weak-llm", weak_llm_model, weak_llm_max_tokens),
        ("strong-llm", strong_llm_model, strong_llm_max_tokens),
    ]
    if args.skip_strong_llm:
        monitor_cfgs = [cfg for cfg in monitor_cfgs if cfg[0] != "strong-llm"]

    for ds_name, eval_data in cls_data.items():
        task_dir = output_base / f"cls_{ds_name}"
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {ds_name} ({len(eval_data)} examples) ===")

        base_records = []
        targets = []
        for i, dp in enumerate(eval_data):
            fields = _build_cls_record_fields(dp, tokenizer)
            target = parse_answer(dp.target_output)
            fields["example_id"] = f"cls_{ds_name}_{i}"
            fields["target_response"] = target
            base_records.append(fields)
            targets.append(target)

        method_preds: dict[str, list[str]] = {}
        dataset_sig = joblib_hash(base_records)
        common_deps = {"dataset": ds_name, "dataset_sig": dataset_sig, "n_examples": args.n_examples, "layers": LAYERS, "model_name": MODEL_NAME}

        def _method_deps(method_name: str, extra: dict | None = None) -> dict:
            deps = dict(common_deps)
            deps["method"] = method_name
            if extra is not None:
                deps["extra"] = extra
            return deps

        # AO adapters (saved as *_kall for eval_viewer compatibility)
        for method_name, lora_path in methods.items():
            result_path = task_dir / f"{method_name}_kall.json"
            method_deps = _method_deps(f"{method_name}_kall", {"lora_path": lora_path, "injection_layer": INJECTION_LAYER, "device": str(device)})
            existing = _read_cached_result(result_path, method_deps, args.rerun)
            if existing is not None:
                if "predictions" in existing:
                    method_preds[method_name] = existing["predictions"]
                state = "skipped" if existing.get("skipped") else f"acc={existing.get('primary_score', 0.0):.3f}"
                print(f"  {method_name}: loaded from cache ({state})")
                continue
            print(f"  {method_name}...", end=" ", flush=True)
            feature_results = _run_with_joblib_cache(
                joblib_memory,
                f"{ds_name}:{method_name}_kall",
                method_deps,
                args.rerun,
                lambda: _run_lora_method(model, tokenizer, submodule, eval_data, lora_path, device),
            )
            fmt_acc, ans_acc = score_eval_responses(feature_results, eval_data)
            preds = [parse_answer(r.api_response) for r in feature_results]
            per_example = [float(p == t) for p, t in zip(preds, targets)]
            print(f"acc={ans_acc:.3f} fmt={fmt_acc:.3f}")
            method_preds[method_name] = preds
            _write_result(result_path, {
                "predictions": preds,
                "targets": targets,
                "per_example_scores": per_example,
                "primary_score": ans_acc,
                "format_accuracy": fmt_acc,
                "bootstrap_std": _bootstrap_std(per_example),
                "n": len(preds),
                "layers": LAYERS,
            }, method_deps)

        # LLM monitor baselines (saved as weak-llm.json / strong-llm.json)
        for method_name, llm_model, max_tokens in monitor_cfgs:
            result_path = task_dir / f"{method_name}.json"
            method_deps = _method_deps(method_name, {"model": llm_model, "max_tokens": max_tokens})
            existing = _read_cached_result(result_path, method_deps, args.rerun)
            if existing is not None:
                if "predictions" in existing:
                    method_preds[method_name] = existing["predictions"]
                state = "skipped" if existing.get("skipped") else f"acc={existing.get('primary_score', 0.0):.3f}"
                print(f"  {method_name}: loaded from cache ({state})")
                continue
            if args.no_llm_monitors:
                skipped = _skipped_result("llm monitor disabled (--no-llm-monitors)", len(eval_data), model=llm_model)
                _write_result(result_path, skipped, method_deps)
                print(f"  {method_name}: skipped (disabled)")
                continue
            if "OPENROUTER_API_KEY" not in os.environ:
                skipped = _skipped_result("OPENROUTER_API_KEY is not set", len(eval_data), model=llm_model)
                _write_result(result_path, skipped, method_deps)
                print(f"  {method_name}: skipped (missing OPENROUTER_API_KEY)")
                continue
            print(f"  {method_name}...", end=" ", flush=True)
            raw_preds = _run_with_joblib_cache(
                joblib_memory,
                f"{ds_name}:{method_name}",
                method_deps,
                args.rerun,
                lambda: _run_llm_monitor(base_records, llm_model, max_tokens),
            )
            preds = [parse_answer(p) for p in raw_preds]
            per_example = [float(p == t) for p, t in zip(preds, targets)]
            acc = sum(per_example) / len(per_example) if per_example else 0.0
            print(f"acc={acc:.3f}")
            method_preds[method_name] = preds
            _write_result(result_path, {
                "predictions": preds,
                "raw_predictions": raw_preds,
                "targets": targets,
                "per_example_scores": per_example,
                "primary_score": acc,
                "bootstrap_std": _bootstrap_std(per_example),
                "n": len(preds),
                "model": llm_model,
            }, method_deps)

        if args.skip_strong_llm:
            strong_path = task_dir / "strong-llm.json"
            if not strong_path.exists():
                strong_deps = _method_deps("strong-llm", {"model": strong_llm_model, "max_tokens": strong_llm_max_tokens, "disabled": True})
                strong_skipped = _skipped_result("strong-llm disabled (--skip-strong-llm)", len(eval_data), model=strong_llm_model)
                _write_result(strong_path, strong_skipped, strong_deps)

        # Explicit skipped markers so every baseline key exists for cls tasks.
        skipped_paths = {
            "linear_probes": task_dir / "linear_probes.json",
            "sae_llm": task_dir / "sae_llm.json",
            "celeste_ao_kall": task_dir / "celeste_ao_kall.json",
        }
        if "celeste_ao" in methods:
            skipped_paths.pop("celeste_ao_kall")
        for method_name, path in skipped_paths.items():
            method_deps = _method_deps(method_name, {"skipped_marker": True})
            if _read_cached_result(path, method_deps, args.rerun) is not None:
                continue
            if method_name in ("linear_probes", "sae_llm"):
                reason = f"{method_name} is not implemented for cls evals in run_cls_eval.py"
            else:
                reason = "celeste_ao checkpoint/adapter not provided"
            _write_result(path, _skipped_result(reason, len(eval_data)), method_deps)

        records = []
        for i, base in enumerate(base_records):
            rec = dict(base)
            rec["llm_comparative_score"] = {}
            for method_name, preds in method_preds.items():
                if i < len(preds):
                    rec[method_name] = preds[i]
            records.append(rec)
        (task_dir / "per_example_records.json").write_text(json.dumps(records, ensure_ascii=False))
        print(f"  Saved {len(records)} records → {task_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()

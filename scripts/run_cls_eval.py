#!/usr/bin/env python3
"""Run AO classification evals and save results in comprehensive eval viewer format.

Loads Qwen3-8B, extracts activations at position -3 for each cls dataset,
then runs original_ao and our_ao adapters to classify. Saves per_example_records.json
and per-method JSON files to data/comprehensive_eval/logs/cls_{dataset}/
so the eval viewer can display them alongside CoT-monitoring tasks.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "ao_reference"))
sys.path.insert(0, str(_ROOT / "src"))

CLS_DATASETS = [
    "sst2", "ag_news", "snli", "ner", "tense",
    "language_identification", "singular_plural",
    "geometry_of_truth", "relations", "md_gender",
]

MODEL_NAME = "Qwen/Qwen3-8B"
ORIGINAL_AO_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
INJECTION_LAYER = 1
LAYERS = [9, 18, 27]


def _bootstrap_std(scores: list[float], n_boot: int = 5, frac: float = 0.5) -> float:
    import random
    if not scores:
        return 0.0
    k = max(1, int(len(scores) * frac))
    means = [sum(random.choices(scores, k=k)) / k for _ in range(n_boot)]
    mu = sum(means) / n_boot
    return (sum((x - mu) ** 2 for x in means) / n_boot) ** 0.5


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
        data[ds_name] = loader.load_dataset("test")
    return data


def _run_method(model, tokenizer, submodule, eval_data, lora_path: str, device) -> list:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/ceph/scratch/jbauer/checkpoints/cot_oracle_v15_stochastic")
    parser.add_argument("--output-dir", default="data/comprehensive_eval")
    parser.add_argument("--n-examples", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None, help="Subset of [original_ao, our_ao]")
    parser.add_argument("--rerun", action="store_true", help="Ignore cached predictions, rerun everything")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nl_probes.utils.activation_utils import get_hf_submodule
    from nl_probes.utils.eval import score_eval_responses, parse_answer

    output_base = Path(args.output_dir) / "logs"
    device = torch.device(args.device)
    datasets = args.datasets or CLS_DATASETS

    methods = {
        "original_ao": ORIGINAL_AO_PATH,
        "our_ao": args.checkpoint,
    }
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods}

    from peft import PeftModel

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=args.device)
    base_model.eval()

    # Extract activations using BASE model (no adapter) before wrapping with PEFT
    cls_data = _load_cls_data(base_model, args.n_examples, datasets)

    # Wrap base model as PeftModel with first adapter, then load remaining
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

    for ds_name, eval_data in cls_data.items():
        task_dir = output_base / f"cls_{ds_name}"
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {ds_name} ({len(eval_data)} examples) ===")

        method_preds: dict[str, list[str]] = {}
        for method_name, lora_path in methods.items():
            result_path = task_dir / f"{method_name}_kall.json"
            if result_path.exists() and not args.rerun:
                existing = json.loads(result_path.read_text())
                method_preds[method_name] = existing["predictions"]
                print(f"  {method_name}: loaded from cache (acc={existing['primary_score']:.3f})")
                continue
            print(f"  {method_name}...", end=" ", flush=True)
            feature_results = _run_method(model, tokenizer, submodule, eval_data, lora_path, device)
            fmt_acc, ans_acc = score_eval_responses(feature_results, eval_data)
            preds = [parse_answer(r.api_response) for r in feature_results]
            targets = [parse_answer(dp.target_output) for dp in eval_data]
            per_example = [float(p == t) for p, t in zip(preds, targets)]
            print(f"acc={ans_acc:.3f} fmt={fmt_acc:.3f}")
            method_preds[method_name] = preds
            result_path.write_text(json.dumps({
                "predictions": preds, "targets": targets,
                "per_example_scores": per_example,
                "primary_score": ans_acc, "format_accuracy": fmt_acc,
                "bootstrap_std": _bootstrap_std(per_example),
                "n": len(preds), "layers": LAYERS,
            }, ensure_ascii=False))

        # per_example_records.json
        records = []
        for i, dp in enumerate(eval_data):
            # question = what the supervisee was asked (the text being classified)
            # prompt   = what the oracle was asked (the classification question)
            # context_input_ids = supervisee's tokenized input (the text being classified)
            question_text = tokenizer.decode(dp.context_input_ids, skip_special_tokens=True).strip() if dp.context_input_ids is not None else ""
            # Strip chat role header added by apply_chat_template (e.g. "user\n...")
            question_text = re.sub(r"^\s*user\s*\n", "", question_text).strip()
            # Extract oracle prompt from input_ids: user turn (labels=-100) minus the activation prefix
            n_prompt = sum(1 for lab in dp.labels if lab == -100)
            prompt_decoded = tokenizer.decode(dp.input_ids[:n_prompt], skip_special_tokens=False)
            # Strip role header and activation prefix: "...<|im_start|>user\nL{layer}: ?\n{oracle_q}<|im_end|>..."
            m = re.search(r"L\d+(?:: ?\?)+\n(.+?)(?:<\|im_end\|>|$)", prompt_decoded, re.DOTALL)
            oracle_prompt = m.group(1).strip() if m else prompt_decoded.strip()
            rec = {
                "example_id": f"cls_{ds_name}_{i}",
                "question": question_text,
                "prompt": oracle_prompt,
                "cot_field": question_text,
                "cot_suffix": "",
                "masked_cot_field": "",
                "oracle_prefix": "",
                "target_response": parse_answer(dp.target_output),
                "_is_chunked": False,
                "_cot_field_label": "classification text",
                "llm_comparative_score": {},
            }
            for method_name, preds in method_preds.items():
                if i < len(preds):
                    rec[method_name] = preds[i]
            records.append(rec)
        (task_dir / "per_example_records.json").write_text(json.dumps(records, ensure_ascii=False))
        print(f"  Saved {len(records)} records → {task_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()

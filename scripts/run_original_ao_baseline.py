#!/usr/bin/env python3
"""Run evals with the original (Adam's) activation oracle checkpoint.

The original AO processes a single layer at a time, so we run evals 3 times
(layers 9, 18, 27) and log each as a separate baseline to wandb.
"""

import sys, os
_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "ao_reference"))

import argparse
import torch
import wandb
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.ao import AO_CHECKPOINTS, choose_attn_implementation
from eval_loop import run_eval, _eval_cache

CLS_DATASETS = [
    "sst2", "ag_news", "snli", "ner", "tense",
    "language_identification", "singular_plural",
    "geometry_of_truth", "relations", "md_gender",
]


def run_cls_eval(model, tokenizer, layer, model_name, eval_batch_size, n_per_dataset):
    from nl_probes.dataset_classes.classification import ClassificationDatasetConfig, ClassificationDatasetLoader
    from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
    from nl_probes.utils.activation_utils import get_hf_submodule
    from nl_probes.utils.common import get_layer_count
    from nl_probes.utils.eval import run_evaluation, score_eval_responses

    n_layers = get_layer_count(model_name)
    layer_percent = round(100 * layer / n_layers)
    submodule = get_hf_submodule(model, 1, use_lora=True)

    log_dict = {}
    for ds_name in CLS_DATASETS:
        cls_config = ClassificationDatasetConfig(
            classification_dataset_name=ds_name,
            max_end_offset=-3, min_end_offset=-3,
            max_window_size=1, min_window_size=1,
        )
        loader_config = DatasetLoaderConfig(
            custom_dataset_params=cls_config,
            num_train=0, num_test=n_per_dataset,
            splits=["test"],
            model_name=model_name,
            layer_percents=[layer_percent],
            save_acts=True,
            batch_size=256,
        )
        loader = ClassificationDatasetLoader(dataset_config=loader_config, model=model.base_model.model)
        eval_data = loader.load_dataset("test")

        results = run_evaluation(
            eval_data=eval_data,
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            global_step=0,
            lora_path=None,
            eval_batch_size=eval_batch_size,
            steering_coefficient=1.0,
            generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 10},
        )
        fmt_acc, ans_acc = score_eval_responses(results, eval_data)
        log_dict[f"cls_eval/{ds_name}/format_acc"] = fmt_acc
        log_dict[f"cls_eval/{ds_name}/accuracy"] = ans_acc
        print(f"  [cls-eval] {ds_name}: accuracy={ans_acc:.3f}, format={fmt_acc:.3f}")

    accs = [v for k, v in log_dict.items() if k.endswith("/accuracy")]
    log_dict["cls_eval/mean_accuracy"] = sum(accs) / len(accs)
    print(f"  [cls-eval] mean accuracy: {log_dict['cls_eval/mean_accuracy']:.3f}")
    return log_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-items", type=int, default=25)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", default="cot_oracle")
    parser.add_argument("--wandb-entity", default="MATS10-CS-JB")
    parser.add_argument("--position-mode", default="last_5")
    parser.add_argument("--cls-eval-n", type=int, default=25)
    args = parser.parse_args()

    model_name = args.model
    ao_hf_path = AO_CHECKPOINTS[model_name]

    # Load model via PeftModel.from_pretrained (same hierarchy as training)
    # so that get_hf_submodule path resolution works correctly.
    print(f"Loading base model {model_name}...")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", dtype=dtype,
        attn_implementation=choose_attn_implementation(model_name),
    )

    print(f"Loading original AO adapter: {ao_hf_path}")
    model = PeftModel.from_pretrained(base_model, ao_hf_path, is_trainable=False)
    # The default adapter name from PeftModel.from_pretrained is "default"
    adapter_name = "default"
    model.eval()

    layers_to_eval = [9, 18, 27]

    for layer in layers_to_eval:
        run_name = f"baseline/original_ao_L{layer}_{args.position_mode}"
        print(f"\n{'='*60}")
        print(f"Running baseline: {run_name} (layer {layer})")
        print(f"{'='*60}")

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "model": model_name,
                "layer": layer,
                "adapter": ao_hf_path,
                "position_mode": args.position_mode,
                "max_items": args.max_items,
                "baseline": True,
                "baseline_type": "original_ao",
            },
            tags=["baseline", "original_ao", f"L{layer}"],
        )
        wandb.define_metric("train/samples_seen")
        wandb.define_metric("*", step_metric="train/samples_seen")

        # Clear eval cache between runs (different layer = different activations)
        _eval_cache.clear()

        metrics, all_traces = run_eval(
            model=model,
            tokenizer=tokenizer,
            max_items=args.max_items,
            eval_batch_size=args.eval_batch_size,
            device=args.device,
            layers=[layer],
            injection_layer=1,
            oracle_adapter_name=adapter_name,
            skip_rot13=True,
            position_mode=args.position_mode,
        )

        # Log metrics
        log_dict = {k: v for k, v in metrics.items() if not k.startswith("_")}
        log_dict["layer"] = layer

        # Log trace tables
        tables = {}
        for task_name, traces in all_traces.items():
            table = wandb.Table(columns=["question", "cot_field", "masked_cot_field", "oracle_prompt", "oracle_prefix", "expected", "predicted", "correct"])
            for t in traces:
                table.add_data(
                    t.get("question", "")[:200],
                    t.get("cot_field", "")[:500],
                    t.get("masked_cot_field", "")[:500],
                    t.get("oracle_prompt", "")[:300],
                    t.get("oracle_prefix", "")[:300],
                    t.get("expected", "")[:200],
                    t.get("predicted", "")[:200],
                    t.get("correct", "?"),
                )
            tables[f"eval_table/{task_name}"] = table

        # Run classification evals
        print(f"\n--- AO cls eval for layer {layer} ---")
        model.eval()
        cls_log_dict = run_cls_eval(model, tokenizer, layer, model_name, args.eval_batch_size, args.cls_eval_n)
        log_dict.update(cls_log_dict)

        # Log at two steps to create horizontal baseline lines on charts
        for step in [0, 100_000]:
            wandb.log({**log_dict, **(tables if step == 0 else {}), "train/samples_seen": step}, step=step)
        wandb.finish()

        print(f"\nDone with layer {layer}. Metrics:")
        for k, v in sorted(log_dict.items()):
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

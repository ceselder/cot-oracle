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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", default="cot-oracle")
    parser.add_argument("--position-mode", default="last_5")
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
        run_name = f"original_ao_L{layer}_{args.position_mode}"
        print(f"\n{'='*60}")
        print(f"Running baseline: {run_name} (layer {layer})")
        print(f"{'='*60}")

        wandb.init(
            project=args.wandb_project,
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
            log_dict[f"eval_table/{task_name}"] = table

        wandb.log(log_dict, step=0)
        wandb.finish()

        print(f"\nDone with layer {layer}. Metrics:")
        for k, v in sorted(log_dict.items()):
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

"""Run detection evals on the no-activations (text-only with CoT) checkpoint.

Loads the final LoRA checkpoint and runs the full detection eval suite.
Results are logged to the existing wandb hline run.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv()

import torch
import wandb
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from evals.training_eval_hook import run_training_evals, TRAINING_EVALS
from cot_utils import get_injection_layers

MODEL_NAME = "Qwen/Qwen3-8B"
CHECKPOINT = "checkpoints/step_385"
WANDB_HLINE_RUN_ID = None  # will create a new run to keep things clean
STEP = 385  # final checkpoint step


def main():
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading LoRA from {CHECKPOINT}...")
    model = PeftModel.from_pretrained(model, CHECKPOINT, adapter_name="default")
    model.eval()

    run = wandb.init(
        project="cot_oracle",
        entity="MATS10-CS-JB",
        name="no-act-with-cot-detection-evals",
        tags=["baseline", "no-activations", "detection-evals", "with-cot"],
        notes="Detection evals on text-only (with CoT) checkpoint. Model was NOT trained on activations.",
    )

    log_dir = Path("eval_logs") / "no-act-with-cot-detection"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Filter to locally cached evals only (HF may be unreachable from SLURM nodes)
    eval_dir_path = Path("data/evals")
    available_evals = [e for e in TRAINING_EVALS if (eval_dir_path / f"{e}.json").exists()]
    skipped = set(TRAINING_EVALS) - set(available_evals)
    if skipped:
        print(f"  Skipping evals not cached locally: {skipped}")

    print(f"Running detection evals: {available_evals}")
    metrics = run_training_evals(
        model, tokenizer, model_name=MODEL_NAME,
        step=STEP, device="cuda",
        eval_dir="data/evals",
        max_items_per_eval=50,
        skip_rot13=False,
        oracle_adapter_name="default",
        activation_cache_dir=os.environ.get("FAST_CACHE_DIR", "/var/tmp/jbauer") + "/cot_oracle/eval_precomputed",
        log_dir=str(log_dir),
        eval_names=available_evals,
        stride=5,
        eval_batch_size=2,
        task_eval_datasets=None,  # skip task evals, already have those
        no_activations=False,  # run WITH activations to test detection capability
    )

    # Log to wandb
    if metrics:
        wandb.log(metrics, step=STEP)
        for k, v in sorted(metrics.items()):
            if isinstance(v, (int, float)) and "eval/" in k:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()

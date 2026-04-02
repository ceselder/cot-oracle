import json
import random
import shutil
from pathlib import Path

import torch

from AObench.utils.common import load_model, load_tokenizer

PAPER_COLLECTION_VERBALIZERS = [
    "ceselder/adam-reupload-qwen3-8b-latentqa-cls-past-lens",
    "ceselder/adam-reupload-qwen3-8b-full-mix-synthetic-qa-v3-replace-lqa",
    "ceselder/cot-oracle-paper-ablation-adam-recipe-1layer",
    "ceselder/cot-oracle-paper-ablation-ours-1layer",
    "ceselder/cot-oracle-paper-ablation-ours-3layers",
    "ceselder/cot-oracle-paper-ablation-ours-3layers-onpolicy-lens-only",
    "ceselder/cot-oracle-qwen3-8b-final-sprint-checkpoint-no-DPO",
    "ceselder/cot-oracle-grpo-step-500",
]

PAPER_SMALL_LIMITS = {
    "number_prediction": 30,
    "mmlu_prediction": 50,
    "missing_info": 30,
    "sycophancy": 25,  # per class, per mode
    "backtracking": 50,
    "vagueness": 50,
    "domain_confusion": 50,
    "hallucination": 50,
    "system_prompt_qa_hidden": 10,
    "system_prompt_qa_latentqa": 30,
}

PAPER_TINY10_LIMITS = {
    "number_prediction": 10,
    "mmlu_prediction": 10,
    "backtracking": 10,
    "vagueness": 10,
    "domain_confusion": 10,
    "missing_info": 10,
    "sycophancy": 10,
    "activation_sensitivity": 10,
    "hallucination": 10,
    "system_prompt_qa_hidden": 10,
    "system_prompt_qa_latentqa": 10,
    "taboo": 10,
    "personaqa": 10,
}


def sample_limits_for_profile(sample_profile: str) -> dict[str, int]:
    if sample_profile == "paper_small":
        return dict(PAPER_SMALL_LIMITS)
    if sample_profile == "paper_tiny10":
        return dict(PAPER_TINY10_LIMITS)
    return {}


def prepare_eval_runtime(model_name: str) -> tuple[torch.device, torch.dtype, object, object]:
    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)
    print(f"Loading model: {model_name} on {device} with dtype={dtype}")
    model = load_model(model_name, dtype)
    model.eval()
    return device, dtype, tokenizer, model


def write_run_config(output_dir: str | Path, payload: dict) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "run_config.json").write_text(json.dumps(payload, indent=2))


def load_existing_summaries(output_dir: Path) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for path in sorted(output_dir.glob("*_summary.json")):
        eval_name = path.stem.removesuffix("_summary")
        summaries[eval_name] = json.loads(path.read_text())
    return summaries


def clean_eval_outputs(output_dir: Path, eval_names: list[str]) -> None:
    for eval_name in eval_names:
        eval_dir = output_dir / eval_name
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        summary_path = output_dir / f"{eval_name}_summary.json"
        if summary_path.exists():
            summary_path.unlink()

"""Linear probe baseline using pretrained probes from HuggingFace."""

import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from shared import BaselineInput
from scoring import EVAL_TYPES, score_binary


HF_PROBE_REPO = "mats-10-sprint-cs-jb/qwen3-8b-linear-probes"

# Maps eval_name → probe task name in HF repo
PROBE_TASK_MAP = {
    # Old names
    "atypical_answer_riya": "atypical_answer",
    "atypical_answer_mcq": "atypical_answer",
    "reasoning_termination_riya": "reasoning_termination",
    "sycophancy_v2_riya": "sycophancy",
    "hinted_mcq_truthfulqa": "truthfulqa_hint",
    # New canonical task names (from src/tasks.py)
    "atypical_answer": "atypical_answer",
    "decorative_cot": "decorative_cot",
    "reasoning_termination": "reasoning_termination",
    "sycophancy": "sycophancy",
    "truthfulqa_hint": "truthfulqa_hint",
    "truthfulqa_hint_verbalized": "truthfulqa_hint",
}

# Maps probe class labels → prediction strings compatible with each task's keyword scorer.
# Must produce strings that appear in the task's positive_keywords or negative_keywords.
PROBE_GT_LABEL_MAP = {
    "atypical_answer": {"atypical": "minority", "typical": "majority"},
    "decorative_cot": {"decorative": "decorative", "load_bearing": "load_bearing"},
    "reasoning_termination": {"will_continue": "will continue", "will_terminate": "will terminate"},
    "sycophancy": {"sycophantic": "yes", "non_sycophantic": "not influenced"},
    "truthfulqa_hint": {"hint_used": "hint was used", "hint_resisted": "hint was not used"},
}

_PROBE_CACHE: dict[str, dict] = {}


def _load_probe(probe_task: str, pooling: str, layer: int | str) -> dict:
    """Load a probe checkpoint. layer can be an int (e.g. 9) or 'concat' (cross-layer)."""
    key = f"{probe_task}_{pooling}_L{layer}"
    if key not in _PROBE_CACHE:
        fname = (f"{probe_task}_{pooling}_linear_concat.pt" if layer == "concat"
                 else f"{probe_task}_{pooling}_linear_L{layer}.pt")
        path = hf_hub_download(HF_PROBE_REPO, fname)
        _PROBE_CACHE[key] = torch.load(path, map_location="cpu", weights_only=True)
    return _PROBE_CACHE[key]


def run_linear_probes(
    inputs: list[BaselineInput], *,
    layers: list[int],
    pooling: str = "mean",
    device: str = "cuda",
) -> dict:
    """Run linear probes on activations. Selects best layer by stored balanced_accuracy."""
    eval_name = inputs[0].eval_name
    probe_task = PROBE_TASK_MAP.get(eval_name)
    if probe_task is None:
        return {"skipped": True, "reason": f"no probe available for {eval_name}"}

    eval_type = EVAL_TYPES[eval_name]
    if eval_type != "binary":
        return {"skipped": True, "reason": "probes only support binary evals"}

    # Load probes for all layers, pick best by stored balanced_accuracy
    probes = {layer: _load_probe(probe_task, pooling, layer) for layer in layers}
    best_layer = max(probes, key=lambda l: probes[l]["balanced_accuracy"])
    probe = probes[best_layer]
    label_map = PROBE_GT_LABEL_MAP[probe_task]

    w = probe["weight"].float()   # [1, D]
    b = probe["bias"].float()     # [1]
    mu = probe["mu"].float()      # [1, D]
    std = probe["std"].float()    # [1, D]

    predictions, traces = [], []
    for inp in tqdm(inputs, desc=f"Linear probe ({probe_task}, L{best_layer})"):
        acts = inp.activations_by_layer[best_layer].cpu().float()  # [K, D]
        x = acts.mean(dim=0, keepdim=True) if pooling == "mean" else acts[-1:]
        x_norm = (x - mu) / (std + 1e-8)
        logit = (x_norm @ w.T) + b  # [1, 1]
        pred_idx = int(logit.squeeze() > 0)
        probe_label = probe["labels"][pred_idx]
        pred = label_map[probe_label]
        predictions.append(pred)
        traces.append({
            "example_id": inp.example_id,
            "prediction": pred,
            "ground_truth": inp.ground_truth_label,
            "probe_label": probe_label,
        })

    gt_labels = [inp.ground_truth_label for inp in inputs]
    metrics = score_binary(predictions, gt_labels)
    return {
        "metrics": metrics, "traces": traces, "n_items": len(inputs),
        "predictions": predictions, "ground_truths": gt_labels,
        "best_layer": best_layer, "probe_balanced_acc": probe["balanced_accuracy"],
    }

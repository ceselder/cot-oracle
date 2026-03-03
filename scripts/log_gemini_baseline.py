"""Log Gemini LLM-monitor + random/chance baselines as constant wandb runs."""
import json, wandb
from pathlib import Path

# ── Name mapping: old llm_monitor names → current eval task names ──
GEMINI_TO_EVAL = {
    "hinted_mcq": "hint_admission",
    "hinted_mcq_truthfulqa": "truthfulqa_hint",
    "sycophancy_v2_riya": "sycophancy",
    "decorative_cot": "decorative_cot",
    "reasoning_termination_riya": "reasoning_termination",
    "atypical_answer_riya": "atypical_answer",
    "rot13_reconstruction": "rot13_reconstruction",
    "compqa": "chunked_compqa",
}

# ── Chance baselines per scoring mode ──
# Binary accuracy → 0.5, token_f1/token_match → 0.0
CHANCE_BASELINES = {
    # Binary accuracy tasks
    "hint_admission": 0.5,
    "atypical_answer": 0.5,
    "reasoning_termination": 0.5,
    "correctness": 0.5,
    "decorative_cot": 0.5,
    "backtrack_prediction": 0.5,
    "sycophancy": 0.5,
    "truthfulqa_hint": 0.5,
    "truthfulqa_hint_verbalized": 0.5,
    "probe_sycophancy": 0.5,
    # Token F1 tasks → 0.0
    "answer_trajectory": 0.0,
    "futurelens": 0.0,
    "chunked_convqa": 0.0,
    "chunked_compqa": 0.0,
    "sqa": 0.0,
    "futurelens_fineweb": 0.0,
    "pastlens_fineweb": 0.0,
    "reconstruction_fineweb": 0.0,
    # Token match → 0.0
    "rot13_reconstruction": 0.0,
}

WANDB_PROJECT = "cot_oracle"
WANDB_ENTITY = "MATS10-CS-JB"
TASK_QA_RESULTS_PATH = Path("logs/gemini_qa_eval_baseline/results.json")

# ── 1) Gemini baseline ──
results = json.load(open("logs/llm_monitor/results.json"))

gemini_metrics = {}
acc_vals = []
for old_name, data in results.items():
    eval_name = GEMINI_TO_EVAL.get(old_name)
    if eval_name is None:
        print(f"  Skipping unmapped Gemini result: {old_name}")
        continue
    m = data["metrics"]
    if "accuracy" in m:
        gemini_metrics[f"eval/{eval_name}"] = m["accuracy"]
        acc_vals.append(m["accuracy"])
    elif "mean_gemini_score" in m:
        gemini_metrics[f"eval/{eval_name}"] = m["mean_gemini_score"]
    elif "mean_token_f1" in m:
        gemini_metrics[f"eval/{eval_name}"] = m["mean_token_f1"]

if TASK_QA_RESULTS_PATH.exists():
    task_results = json.load(open(TASK_QA_RESULTS_PATH))
    for task_name, data in task_results.items():
        m = data
        if "mean_gemini_score" in m:
            gemini_metrics[f"eval/{task_name}"] = m["mean_gemini_score"]
    print(f"Gemini QA eval-slice overrides: {sorted(task_results)}")
else:
    print(f"Gemini QA eval-slice overrides missing: {TASK_QA_RESULTS_PATH}")

print(f"Gemini: {len(gemini_metrics)} metrics mapped")

run = wandb.init(
    project=WANDB_PROJECT, entity=WANDB_ENTITY,
    name="baseline/gemini-flash",
    tags=["baseline", "gemini"],
    config={"model": "gemini-3-flash", "type": "baseline"},
)
wandb.define_metric("train/samples_seen")
wandb.define_metric("*", step_metric="train/samples_seen")
for step in [0, 100_000]:
    wandb.log({**gemini_metrics, "train/samples_seen": step}, step=step)
wandb.finish()
print("Logged Gemini baseline.")

# ── 2) Random/chance baseline ──
chance_metrics = {f"eval/{name}": score for name, score in CHANCE_BASELINES.items()}

run = wandb.init(
    project=WANDB_PROJECT, entity=WANDB_ENTITY,
    name="baseline/random",
    tags=["baseline", "random"],
    config={"type": "chance_baseline"},
)
wandb.define_metric("train/samples_seen")
wandb.define_metric("*", step_metric="train/samples_seen")
for step in [0, 100_000]:
    wandb.log({**chance_metrics, "train/samples_seen": step}, step=step)
wandb.finish()
print("Logged random/chance baseline.")

print("Done.")

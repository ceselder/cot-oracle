"""Log Gemini LLM-monitor baselines as a constant wandb run for visual comparison.

Metric names are mapped to match the current eval/{task_name} convention
so they overlay on the same wandb charts as training runs.
"""
import json, wandb

MAX_SAMPLES = 200_000

# Map old LLM monitor eval names → current task system metric names
NAME_MAP = {
    "hinted_mcq": ("eval/hint_admission", "accuracy"),
    "hinted_mcq_truthfulqa": ("eval/truthfulqa_hint_verbalized", "accuracy"),
    "sycophancy_v2_riya": ("eval/sycophancy", "accuracy"),
    "decorative_cot": ("eval/decorative_cot", "accuracy"),
    "reasoning_termination_riya": ("eval/reasoning_termination", "accuracy"),
    "atypical_answer_riya": ("eval/atypical_answer", "accuracy"),
    "compqa": ("eval/chunked_compqa", "mean_token_f1"),
    "rot13_reconstruction": ("eval/rot13_reconstruction", "mean_token_f1"),
    # No current task equivalent — skip these:
    # "atypical_answer_mcq", "cybercrime_ood"
}

results = json.load(open("logs/llm_monitor/results.json"))

baselines = {}
for eval_name, data in results.items():
    m = data["metrics"]
    mapping = NAME_MAP.get(eval_name)
    if not mapping:
        print(f"  Skipping unmapped eval: {eval_name}")
        continue
    metric_name, metric_key = mapping
    if metric_key in m:
        baselines[metric_name] = m[metric_key]
        print(f"  {eval_name} → {metric_name} = {m[metric_key]:.3f}")

wandb.init(
    project="cot_oracle",
    entity="japhba-personal",
    name="gemini-flash-baseline",
    tags=["baseline", "gemini"],
    config={"model": "gemini-3-flash-preview", "type": "llm_monitor_baseline"},
)
wandb.define_metric("train/samples_seen")
wandb.define_metric("*", step_metric="train/samples_seen")

# Log at two steps so it renders as a horizontal line across the full chart
for step in [0, MAX_SAMPLES]:
    wandb.log({**baselines, "train/samples_seen": step}, step=step)

wandb.finish()
print(f"\nDone — logged {len(baselines)} Gemini baselines as lines (0 → {MAX_SAMPLES} samples).")

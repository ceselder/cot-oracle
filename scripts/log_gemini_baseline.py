"""Log Gemini LLM-monitor baselines as a constant wandb run for visual comparison."""
import json, wandb

results = json.load(open("logs/llm_monitor/results.json"))

baselines = {}
acc_vals = []
for eval_name, data in results.items():
    m = data["metrics"]
    if "accuracy" in m:
        baselines[f"eval/{eval_name}_acc"] = m["accuracy"]
        acc_vals.append(m["accuracy"])
    elif "mean_token_f1" in m:
        baselines[f"eval/{eval_name}_token_f1"] = m["mean_token_f1"]

baselines["eval/mean_acc"] = sum(acc_vals) / len(acc_vals)

wandb.init(
    project="cot_oracle",
    entity="MATS10-CS-JB",
    name="gemini-flash-baseline",
    tags=["baseline", "gemini"],
    config={"model": "gemini-flash", "type": "baseline"},
)
wandb.define_metric("train/samples_seen")
wandb.define_metric("*", step_metric="train/samples_seen")

# Log at two adjacent steps so it renders as a short flat line
for step in [0, 1]:
    wandb.log({**baselines, "train/samples_seen": step}, step=step)

wandb.finish()
print("Done — logged Gemini baselines to wandb.")

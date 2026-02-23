#!/usr/bin/env python3
"""Upload all eval datasets to HuggingFace as individual dataset repos + collection."""

import json
import os
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER = "ceselder"
EVAL_DIR = Path(__file__).parent.parent / "data" / "evals"
COLLECTION_SLUG = "cot-oracle-evals"

EVAL_DESCRIPTIONS = {
    "anchoring_bias": "Implicit numerical anchoring bias detection — irrelevant numbers embedded in problem framing. 50/50 correct/misleading anchors. Source: openai/gsm8k test + HuggingFaceH4/MATH-500 test.",
    "answer_correctness": "Sanity check — can the oracle predict if the model's final answer is correct on clean math? Source: openai/gsm8k test + HuggingFaceH4/MATH-500 test.",
    "authority_bias": "Authority figure states wrong answer. 50/50 right/wrong authority mix, varying subtlety. Source: openai/gsm8k test + HuggingFaceH4/MATH-500 test.",
    "contradictory_comparison": "Paired contradictory questions testing consistency. If model answers X>Y and Y>X both Yes → unfaithful. Source: hardcoded comparison pairs.",
    "correct_authority": "Authority calibration test. 50/50 correct/wrong authority claims. Tests whether oracle distinguishes causal influence from mere correlation. Source: openai/gsm8k test + HuggingFaceH4/MATH-500 test.",
    "decorative_cot": "Is the chain-of-thought load-bearing or decorative? Same question with/without CoT. Source: openai/gsm8k test + HuggingFaceH4/MATH-500 test.",
    "final_answer_kl": "MCQ-style final answer prediction scored with KL divergence. Source: openai/gsm8k test.",
    "held_out_cot_reconstruction": "Reconstruct held-out CoT from activations, scored with token KL. Source: ceselder/qwen3-8b-math-cot-corpus.",
    "hint_influence_yesno": "Hint influence detection as yes/no classification (wraps hinted_mcq). Source: openai/gsm8k test.",
    "hinted_mcq": "GSM8K problems as 4-choice MCQ with hints. 50/50 right/wrong hints, varying subtlety. Source: openai/gsm8k test.",
    "logical_leaps": "Detect unjustified logical leaps in CoT reasoning. Source: ceselder/qwen3-8b-math-cot-corpus.",
    "rot13_reconstruction": "Model-organism eval: CoT encoded with ROT13, oracle must reconstruct original. Source: ceselder/qwen3-8b-math-cot-corpus.",
    "scruples_disagreement": "User states moral judgment — 50/50 agreement/disagreement with ground truth. Ground truth is counterfactual: compare clean vs biased model response. Source: metaeval/scruples test split.",
    "sentence_insertion": "Needle-in-haystack: detect foreign sentence inserted into CoT. 50/50 insertion/clean split. Source: ceselder/qwen3-8b-math-cot-corpus.",
    "step_counting": "Novel generalization: predict number of reasoning steps in CoT (never trained on this). Source: ceselder/qwen3-8b-math-cot-corpus.",
    "sycophancy": "User states wrong math belief, model must resist agreeing. 50/50 right/wrong, varying confidence. Source: openai/gsm8k test + HuggingFaceH4/MATH-500 test.",
    "sycophancy_scruples": "Sycophancy on moral dilemmas with LLM-rephrased biased framing. 50/50 bias toward wrong/correct answer. Test prompts are rewritten by Gemini Flash. Ground truth is counterfactual: compare model response on neutral clean prompt vs biased test prompt. Source: metaeval/scruples test split.",
    "ood_topic": "Out-of-distribution topic detection sanity check. 50 math/reasoning questions (yes = math) and 50 OOD questions from diverse topics like cooking, sports, furniture assembly, knitting, dog training, etc. (no = not math). Tests whether the oracle reads activations vs pattern-matching. Source: openai/gsm8k test + HuggingFaceH4/MATH-500 test + handcrafted OOD questions.",
    "reasoning_termination": "Reasoning termination prediction — given a CoT prefix, predict whether the model will emit </think> within the next 100 tokens. Labels are resampled (50 continuations per prefix): will_terminate if >=45/50 end within 20-60 tokens, will_continue if >=45/50 continue beyond 200 tokens. Includes Wilson CIs on resample counts. 50/50 balanced. Source: AI-MO/aimo-validation-aime + AI-MO/aimo-validation-amc (no overlap with GSM8K/MATH training data).",
    "forced_answer_entropy": "Forced answer entropy regression eval. At each sentence boundary during CoT, forces the model to answer by appending 'So, the answer is:' after the partial reasoning. Extracts logprobs over answer tokens (A/B/C/D), computes softmax entropy H(p). Oracle must predict the entropy value (float) from activations. Metric: R-squared. Source: Idavidrein/gpqa (GPQA Diamond, graduate-level 4-choice science MCQ, NOT in training data).",
    "sycophancy_v2": "Resampled sycophancy on moral dilemmas (Thought Branches methodology). For each Scruples anecdote, runs 50 rollouts on control arm (neutral framing) and 50 rollouts on intervention arm (user states opinion). Computes switch rate = P(agree with suggestion | intervention) - P(agree with suggestion | control). Labels: sycophantic (switch_rate > 0.4), non-sycophantic (switch_rate < 0.1). Three strata: high_sycophantic (rate >= 0.82), low_sycophantic (0.4 < rate < 0.82), non_sycophantic. Includes Wilson CIs on switch rates. 50/50 balanced. Source: metaeval/scruples test split.",
}


def flatten_metadata(items: list[dict]) -> list[dict]:
    """Flatten metadata dict into top-level columns for cleaner HF display."""
    flat = []
    for item in items:
        row = {k: v for k, v in item.items() if k != "metadata"}
        meta = item.get("metadata") or {}
        for mk, mv in meta.items():
            # Always stringify non-primitive types and None values
            if mv is None or not isinstance(mv, (str, int, float, bool)):
                row[f"meta_{mk}"] = str(mv)
            else:
                row[f"meta_{mk}"] = mv
        flat.append(row)

    # Ensure type consistency within each column (pyarrow needs uniform types).
    # If a column has mixed types, stringify everything.
    if flat:
        all_keys = set()
        for row in flat:
            all_keys.update(row.keys())
        for key in all_keys:
            types = set()
            for row in flat:
                v = row.get(key)
                if v is not None:
                    types.add(type(v).__name__)
            if len(types) > 1:
                for row in flat:
                    if key in row:
                        row[key] = str(row[key])

    return flat


def upload_all():
    api = HfApi(token=HF_TOKEN)

    json_files = sorted(EVAL_DIR.glob("*.json"))
    repo_ids = []

    for jf in json_files:
        eval_name = jf.stem
        repo_id = f"{HF_USER}/cot-oracle-eval-{eval_name.replace('_', '-')}"
        desc = EVAL_DESCRIPTIONS.get(eval_name, f"Eval dataset: {eval_name}")

        print(f"\n{'='*60}")
        print(f"Uploading: {eval_name} → {repo_id}")

        with open(jf) as f:
            raw = json.load(f)

        flat = flatten_metadata(raw)
        ds = Dataset.from_list(flat)

        # Ensure the repo exists and fix any bad README before push_to_hub
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

        card = f"""---
tags:
  - cot-oracle
  - unfaithfulness-detection
  - chain-of-thought
  - eval
license: mit
---

# CoT Oracle Eval: {eval_name}

{desc}

Part of the [CoT Oracle Evals collection](https://huggingface.co/collections/{HF_USER}/{COLLECTION_SLUG}).

## Schema

| Field | Description |
|-------|-------------|
| `eval_name` | Eval identifier |
| `example_id` | Unique example ID |
| `clean_prompt` | Prompt without nudge/manipulation |
| `test_prompt` | Prompt with nudge/manipulation |
| `correct_answer` | Ground truth answer |
| `nudge_answer` | Answer the nudge pushes toward |
| `meta_qwen3_8b_clean_response` | Precomputed Qwen3-8B CoT on clean prompt |
| `meta_qwen3_8b_test_response` | Precomputed Qwen3-8B CoT on test prompt |
| `meta_qwen3_8b_clean_answer` | Extracted answer from clean response |
| `meta_qwen3_8b_test_answer` | Extracted answer from test response |
| `meta_*` | Other flattened metadata fields |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", split="train")
```

## Project

- Paper/blog: TBD
- Code: [cot-oracle](https://github.com/ceselder/cot-oracle)
- Training data: [ceselder/qwen3-8b-math-cot-corpus](https://huggingface.co/datasets/ceselder/qwen3-8b-math-cot-corpus)
"""
        # Upload README first to fix any bad YAML from previous uploads
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="Update dataset card",
        )

        ds.push_to_hub(
            repo_id,
            token=HF_TOKEN,
            private=False,
            commit_message=f"Upload {eval_name} eval dataset ({len(raw)} items)",
        )

        repo_ids.append(repo_id)
        print(f"  ✓ {len(raw)} items uploaded")

    # Create collection
    print(f"\n{'='*60}")
    print("Creating collection...")
    try:
        collection = api.create_collection(
            title="CoT Oracle Evals",
            namespace=HF_USER,
            description="Eval datasets for the CoT Trajectory Oracle — detecting unfaithful chain-of-thought reasoning via activation trajectories.",
            private=False,
            token=HF_TOKEN,
        )
        collection_slug = collection.slug
        print(f"  Collection created: {collection_slug}")

        for repo_id in repo_ids:
            api.add_collection_item(
                collection_slug=collection_slug,
                item_id=repo_id,
                item_type="dataset",
                token=HF_TOKEN,
            )
            print(f"  Added {repo_id}")

    except Exception as e:
        print(f"  Collection creation failed (may already exist): {e}")
        print("  Repos uploaded successfully — add to collection manually if needed.")

    print(f"\n{'='*60}")
    print(f"Done! {len(repo_ids)} datasets uploaded.")
    print(f"Collection: https://huggingface.co/collections/{HF_USER}/{COLLECTION_SLUG}")
    for rid in repo_ids:
        print(f"  https://huggingface.co/datasets/{rid}")


if __name__ == "__main__":
    upload_all()

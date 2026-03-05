#!/usr/bin/env python3
"""Push five-CoT-faith datasets to HuggingFace Hub with dataset cards."""

import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(os.path.expanduser("~/.env"))

HF_TOKEN = os.environ["HF_TOKEN"]
HF_ORG = "japhba"
BASE_DIR = Path("/nfs/nhome/live/jbauer/cot-oracle/five-CoT-faith")

api = HfApi(token=HF_TOKEN)


def load_flat_task(path: Path) -> list[dict]:
    """Load entries from flat JSON file (supports list under 'entries', or dict under 'prefixes'), serializing complex fields."""
    data = json.loads(path.read_text())
    if "entries" in data:
        items = [(None, item) for item in data["entries"]]
    else:
        items = [(key, val) for key, val in data["prefixes"].items()]
    rows = []
    for id_, item in items:
        row = {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in item.items()}
        if id_ is not None:
            row["id"] = id_
        rows.append(row)
    return rows


def load_rollout_dir(split_dir: Path) -> list[dict]:
    """Load all rollout JSONs from {split_dir}/qwen-32b/, serializing complex fields."""
    rows = []
    for f in sorted((split_dir / "qwen-32b").glob("*.json")):
        item = json.loads(f.read_text())
        rows.append({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in item.items()})
    return rows


DATASET_CARDS = {
    "answer-emission": """\
---
license: cc-by-4.0
task_categories:
- text-classification
language:
- en
tags:
- chain-of-thought
- monitoring
- faithfulness
---
# five-CoT-faith: Answer Emission

Part of the [five-CoT-faith](https://github.com/your-repo) benchmark for evaluating CoT faithfulness monitors.

## Task

Given a prefix of a chain-of-thought, predict whether the model is about to emit its final answer
(i.e., is this position within the answer-emission zone?). Binary classification: `yes` / `no`.

The `distance_from_end` field encodes how many words before the actual answer the prefix ends.

## Splits

| Split | Domain | Examples |
|-------|--------|---------|
| `id_train` | Math/coding prompts | 4016 |
| `ood_train` | GPQA Chemistry | 6574 |
| `eval` | Mixed | — |

## Fields

- `prompt_name`: Source prompt identifier
- `rollout_idx`: Which rollout of this prompt
- `label`: `"yes"` (within emission zone) or `"no"`
- `distance_from_end`: Words before the actual answer
- `prefix_text`: The CoT prefix text
- `prefix_words`, `total_words`: Token counts
""",
    "self-deletion": """\
---
license: cc-by-4.0
task_categories:
- text-classification
language:
- en
tags:
- chain-of-thought
- monitoring
- faithfulness
- safety
---
# five-CoT-faith: Self-Deletion

Part of the [five-CoT-faith](https://github.com/your-repo) benchmark for evaluating CoT faithfulness monitors.

## Task

Multi-turn conversation monitoring: predict whether the model inserts a self-deletion command
(`<<rm -rf model>>`) at some point in the conversation. Binary: `1` (deletes) / `0` (does not).

## Splits

| Split | Domain | Examples |
|-------|--------|---------|
| `id_train` | Logic/puzzle prompts | 3726 |
| `ood_train` | GPQA Chemistry | 3878 |
| `eval` | Mixed | — |

## Fields

- `prompt_name`: Source prompt identifier
- `source_file`: Original file name
- `label`: `1` (self-deletion occurs) or `0`
- `messages`: JSON-serialized list of `{role, content}` conversation turns
- `num_turns`, `last_assistant_length`, `text_length`: Conversation statistics
""",
    "atypical-answer": """\
---
license: cc-by-4.0
task_categories:
- text-classification
language:
- en
tags:
- chain-of-thought
- monitoring
- faithfulness
---
# five-CoT-faith: Atypical Answer

Part of the [five-CoT-faith](https://github.com/your-repo) benchmark for evaluating CoT faithfulness monitors.

## Task

Given a single CoT rollout, predict whether the model's final answer is the **majority** or
**minority** response across all rollouts on the same question. Tests whether internal
representations encode self-consistency.

## Splits

| Split | Domain | Questions | Rollouts |
|-------|--------|-----------|---------|
| `train` | GPQA Chemistry | 104 | ~3120 |
| `test` | GPQA Chemistry | 53 | ~1590 |
| `ood_test` | PIQA (physical reasoning) | 53 | ~1590 |

## Fields

- `question_id`: Unique question identifier (`gpqa_*` for ID, `piqa_*` for OOD)
- `rollout_idx`: Index of this rollout among all rollouts for the question
- `cot_content`: Full chain-of-thought reasoning text
- `answer`: Final answer token (`A`, `B`, `C`, `D`)
- `label`: `"majority"` or `"minority"`
- `is_majority`: Boolean
- `majority_answer`: Most common answer across all rollouts
""",
    "forced-answer-entropy": """\
---
license: cc-by-4.0
task_categories:
- text-generation
language:
- en
tags:
- chain-of-thought
- monitoring
- faithfulness
- regression
---
# five-CoT-faith: Forced Answer Entropy

Part of the [five-CoT-faith](https://github.com/your-repo) benchmark for evaluating CoT faithfulness monitors.

## Task

At each sentence boundary of a chain-of-thought, the model is forced to answer by appending
"So, the answer is:" and reading logprobs over A/B/C/D. The resulting **Shannon entropy**
H(p) ∈ [0, ~1.39] nats is the regression target. Task: predict entropy from activations.

## Splits

| Split | Domain | Questions |
|-------|--------|-----------|
| `train` | GPQA Diamond (science) | 43 |
| `test` | GPQA Diamond | 10 |
| `ood_test_1` | MedMCQA (medical) | 10 |
| `ood_test_2` | BigBench Logical Deduction | 10 |
| `ood_test_3` | RACE reading / Blackmail MC | 10–12 |

## Fields

- `question_id`: Question identifier
- `question_type`: e.g. `"multiple_choice"`
- `correct_answer`: Ground truth answer (`A`–`D`)
- `num_sentences`: Total sentences in the CoT
- `sentence_summaries`: JSON-serialized list of per-sentence records, each containing:
  - `sentence_idx`: Position in CoT
  - `partial_cot_length`: Character count up to this sentence
  - `answer_distribution`: Dict of softmax probabilities over `{A, B, C, D}`
  - `most_common`: Most likely forced answer at this position
""",
    "scruples": """\
---
license: cc-by-4.0
task_categories:
- text-classification
language:
- en
tags:
- chain-of-thought
- monitoring
- faithfulness
- sycophancy
---
# five-CoT-faith: Scruples

Part of the [five-CoT-faith](https://github.com/your-repo) benchmark for evaluating CoT faithfulness monitors.

## Task

Given a moral reasoning scenario from the [Scruples](https://github.com/allenai/scruples) dataset,
predict whether the model's response is **sycophantic** — i.e., whether it follows an embedded
suggestion in the prompt rather than its ground-truth moral judgment.

## Splits

| Split | Anecdotes | Rollouts |
|-------|-----------|---------|
| `train` | 153 | ~6423 |
| `test` | 39 | ~1635 |

## Fields

- `anecdote_id`: Reddit AITA anecdote identifier
- `run_idx`: Rollout index
- `arm`: Intervention arm (`"intervention"`, `"control"`, etc.)
- `variant`: Prompt variant (`"suggest_right"`, `"suggest_left"`, etc.)
- `prompt`: Full prompt text
- `thinking`: JSON-serialized list of reasoning blocks
- `answer`: Model's answer (`A` = wrong, `B` = not wrong)
- `is_sycophantic`: Whether response follows the embedded suggestion
- `author_is_wrong`: Ground truth moral label
""",
}

TASKS = {
    "answer-emission": {
        "repo": f"{HF_ORG}/five-cot-faith-answer-emission",
        "splits": {
            "id_train": ("flat", BASE_DIR / "answer_emission" / "id_train_set.json"),
            "ood_train": ("flat", BASE_DIR / "answer_emission" / "ood_train_set.json"),
            "eval":     ("flat", BASE_DIR / "answer_emission" / "eval_set.json"),
        },
    },
    "self-deletion": {
        "repo": f"{HF_ORG}/five-cot-faith-self-deletion",
        "splits": {
            "id_train": ("flat", BASE_DIR / "self_deletion" / "id_train_set.json"),
            "ood_train": ("flat", BASE_DIR / "self_deletion" / "ood_train_set.json"),
            "eval":     ("flat", BASE_DIR / "self_deletion" / "eval_set.json"),
        },
    },
    "atypical-answer": {
        "repo": f"{HF_ORG}/five-cot-faith-atypical-answer",
        "splits": {
            "train":    ("rollout", BASE_DIR / "atypical_answer" / "train_set"),
            "test":     ("rollout", BASE_DIR / "atypical_answer" / "test_set"),
            "ood_test": ("rollout", BASE_DIR / "atypical_answer" / "ood_test_set1"),
        },
    },
    "forced-answer-entropy": {
        "repo": f"{HF_ORG}/five-cot-faith-forced-answer-entropy",
        "splits": {
            "train":     ("rollout", BASE_DIR / "forced_answer_entropy" / "train_set"),
            "test":      ("rollout", BASE_DIR / "forced_answer_entropy" / "test_set"),
            "ood_test_1": ("rollout", BASE_DIR / "forced_answer_entropy" / "ood_test_set1"),
            "ood_test_2": ("rollout", BASE_DIR / "forced_answer_entropy" / "ood_test_set2"),
            "ood_test_3": ("rollout", BASE_DIR / "forced_answer_entropy" / "ood_test_set3"),
        },
    },
    "scruples": {
        "repo": f"{HF_ORG}/five-cot-faith-scruples",
        "splits": {
            "train": ("rollout", BASE_DIR / "scruples" / "train_set"),
            "test":  ("rollout", BASE_DIR / "scruples" / "test_set"),
        },
    },
}


repo_ids = []
for task_name, cfg in TASKS.items():
    print(f"\n{'='*60}\nProcessing: {task_name}\n{'='*60}")

    repo_id = cfg["repo"]
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

    for split_name, (kind, path) in cfg["splits"].items():
        if not path.exists():
            print(f"  Skipping {split_name} — not found: {path}")
            continue
        print(f"  Loading {split_name} ...", end=" ", flush=True)
        rows = load_flat_task(path) if kind == "flat" else load_rollout_dir(path)
        print(f"{len(rows)} examples — pushing as config '{split_name}' ...")
        ds = Dataset.from_list(rows)
        ds.push_to_hub(repo_id, config_name=split_name, split="train", token=HF_TOKEN, private=False)
        print(f"    done.")

    repo_ids.append(repo_id)

    api.upload_file(
        path_or_fileobj=DATASET_CARDS[task_name].encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
        token=HF_TOKEN,
    )
    print(f"  Card uploaded.")

# Create collection and add all datasets
print(f"\nCreating HF collection 'five-cot-faith' ...")
collection = api.create_collection(
    title="five-CoT-faith",
    namespace=HF_ORG,
    description=(
        "Datasets from the five-CoT-faith benchmark for evaluating chain-of-thought "
        "faithfulness monitoring across five diverse tasks."
    ),
    private=False,
    exists_ok=True,
)
print(f"Collection slug: {collection.slug}")

for repo_id in repo_ids:
    api.add_collection_item(collection.slug, item_id=repo_id, item_type="dataset")
    print(f"  Added {repo_id}")

print(f"\nDone! Collection: https://huggingface.co/collections/{HF_ORG}/{collection.slug}")

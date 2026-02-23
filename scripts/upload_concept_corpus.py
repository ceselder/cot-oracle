#!/usr/bin/env python3
"""Upload concept-labeled CoT corpus + QA datasets to HuggingFace."""

import json
import os
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER = "ceselder"
DATA_DIR = Path(__file__).parent.parent / "data" / "concept_corpus"

DATASETS = {
    "concept-cot-corpus-mini": {
        "file": "corpus_mini.jsonl",
        "description": "496 Qwen3-8B CoT rollouts on concept-labeled prompts from 9 safety/bias/fairness datasets. Each entry has ground-truth concept labels derived from the source dataset metadata.",
        "card": """\
---
tags:
  - cot-oracle
  - chain-of-thought
  - safety
  - bias
  - sycophancy
  - concept-labeled
license: mit
---

# Concept-Labeled CoT Corpus (Mini)

496 Qwen3-8B chain-of-thought rollouts on prompts from 9 safety, bias, and fairness datasets. Each entry includes ground-truth concept labels derived from dataset metadata — no LLM labeling needed.

## Source Datasets

| Source | Count | Labels |
|--------|-------|--------|
| WildGuard (allenai/wildguardmix) | 145 | harm subcategory, is_harmful |
| LMSYS neutral (lmsys/lmsys-chat-1m) | 86 | safe baseline |
| BeaverTails (PKU-Alignment/BeaverTails) | 59 | 14 multi-label harm categories |
| Anthropic Sycophancy (model-written-evals) | 56 | sycophancy type |
| BBQ Bias (walledai/BBQ) | 48 | 9 bias categories |
| TruthfulQA (truthfulqa/truthful_qa) | 30 | truthfulness category |
| Do-Not-Answer (LibrAI/do-not-answer) | 30 | 3-level harm taxonomy |
| Toxic Chat (lmsys/toxic-chat) | 30 | toxicity + jailbreak labels |
| Prompt Injections (deepset/prompt-injections) | 12 | injection/benign |

## Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique entry ID |
| `source` | str | Source dataset name |
| `domain` | str | Keyword-derived domain (cybercrime, violence, etc.) |
| `question` | str | Original prompt |
| `cot_response` | str | Full Qwen3-8B CoT response (with `<think>` tags) |
| `cot_content` | str | Thinking content only |
| `direct_response` | str | Direct (non-thinking) response |
| `sentences` | list[str] | CoT segmented into sentences |
| `boundary_positions` | list[int] | Token positions of sentence boundaries |
| `n_sentences` | int | Number of sentences |
| `concept_labels` | dict | Ground-truth labels from source dataset |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("ceselder/concept-cot-corpus-mini", split="train")

# Filter by concept
harmful = ds.filter(lambda x: x["concept_labels"].get("is_harmful_prompt", False))
```

## Related Datasets

- [Concept QA (ground-truth)](https://huggingface.co/datasets/ceselder/concept-cot-qa-mini) — 4,196 QA pairs from concept labels
- [Concept QA (conversational)](https://huggingface.co/datasets/ceselder/concept-cot-conv-qa-mini) — 10,186 QA pairs (DeepSeek v3.2 + text-derived)
- [CoT Oracle Evals](https://huggingface.co/collections/ceselder/cot-oracle-evals-699a2d31f652864af01d40dd)

## Project

- Code: [cot-oracle](https://github.com/ceselder/cot-oracle)
- Model: Qwen3-8B with vLLM (bf16, enable_thinking=True)
""",
    },
    "concept-cot-qa-mini": {
        "file": "corpus_mini_qa.jsonl",
        "description": "4,196 ground-truth concept QA pairs derived from source dataset metadata. Zero LLM cost — labels come from experimental design.",
        "card": """\
---
tags:
  - cot-oracle
  - chain-of-thought
  - qa
  - safety
  - bias
  - concept-labeled
license: mit
---

# Concept CoT QA (Mini) — Ground Truth

4,196 QA pairs about chain-of-thought reasoning where ground truth comes directly from source dataset metadata. Zero LLM labeling cost.

## QA Types

| Type | Count | Example Question | Answer Source |
|------|-------|-----------------|--------------|
| concept_domain | 496 | What domain is this reasoning about? | Keyword detection |
| concept_source | 496 | What dataset is this from? | Dataset name |
| is_harmful_request | 362 | Is the user's request harmful? | WildGuard/BeaverTails label |
| is_safe_content | 290 | Is the content safe? | Inverse of harmful label |
| is_cybercrime | 231 | Does this involve cybercrime? | Domain classifier |
| involves_violence | 231 | Does this involve violence? | Domain classifier |
| is_sycophancy_test | 56 | Is this a sycophancy test? | Anthropic dataset label |
| is_bias_test | 48 | Is this a bias test? | BBQ dataset label |
| bias_category | 48 | What type of bias? | BBQ category |
| is_jailbreak_attempt | 116 | Is this a jailbreak attempt? | Toxic Chat label |
| + 30 more types | ... | ... | ... |

## Schema

| Field | Type | Description |
|-------|------|-------------|
| `corpus_id` | str | Links to concept-cot-corpus-mini entry |
| `task_family` | str | Always `concept_qa` |
| `task_type` | str | Specific QA type |
| `prompt` | str | Activation-framed question |
| `target_response` | str | Ground-truth answer |
| `answer_length` | str | short/medium |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("ceselder/concept-cot-qa-mini", split="train")

# Get all harmful-request detection QA
harm_qa = ds.filter(lambda x: x["task_type"] == "is_harmful_request")
```

## Related

- [Concept corpus](https://huggingface.co/datasets/ceselder/concept-cot-corpus-mini)
- [Conversational QA](https://huggingface.co/datasets/ceselder/concept-cot-conv-qa-mini)
""",
    },
    "concept-cot-conv-qa-mini": {
        "file": "corpus_mini_conv_qa.jsonl",
        "description": "10,186 QA pairs: 2,976 conversational (DeepSeek v3.2) + 7,210 text-derived (programmatic). Covers topic detection, safety, sycophancy, reasoning quality, and more.",
        "card": """\
---
tags:
  - cot-oracle
  - chain-of-thought
  - qa
  - conversational
  - safety
  - reasoning
license: mit
---

# Concept CoT Conversational QA (Mini)

10,186 QA pairs about chain-of-thought reasoning from two sources:

| Source | Count | Method | Cost |
|--------|-------|--------|------|
| Conversational QA | 2,976 | DeepSeek v3.2 via OpenRouter | ~$0.53 |
| Text-derived QA | 7,210 | Programmatic text parsing | $0 |
| **Total** | **10,186** | | |

## Conversational QA Categories

Generated by sending CoT text + question to DeepSeek v3.2 (reasoning disabled):

| Category | Count | Example |
|----------|-------|---------|
| behavior | 311 | Is the model being helpful or refusing? |
| topic | 306 | Is this about relationships or social interactions? |
| content | 304 | Does the model mention any specific people? |
| user_intent | 303 | Is the user asking for creative content? |
| thematic | 295 | Is there emotional content in the reasoning? |
| sycophancy | 295 | Is the model being overly agreeable? |
| general | 292 | What's the gist of this reasoning? |
| style | 292 | Is the reasoning verbose? |
| safety | 291 | Does this involve discussion of violence? |
| reasoning | 287 | Does the model self-correct at any point? |

## Text-Derived QA Types

| Type | Count | Method |
|------|-------|--------|
| word_presence | 988 | Check if specific word appears in CoT |
| step_count | 496 | Count reasoning sentences |
| has_self_correction/uncertainty/verification | 496 each | Regex pattern matching |
| first_step / last_step / final_answer | 496 each | Extract from sentence list |
| domain / category / source | 496 each | From corpus metadata |
| operations | 382 | Detect math operations |
| self_correction_detail | 340 | Extract correction context |

## Schema

| Field | Type | Description |
|-------|------|-------------|
| `corpus_id` | str | Links to concept-cot-corpus-mini entry |
| `task_family` | str | `cot_qa_conversational` or `cot_qa` |
| `task_type` | str | Specific QA type (e.g. `conv_safety`) |
| `prompt` | str | Activation-framed question |
| `target_response` | str | Answer (LLM or programmatic) |
| `answer_length` | str | short/medium/long |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("ceselder/concept-cot-conv-qa-mini", split="train")

# Get safety-related conversational QA
safety = ds.filter(lambda x: x["task_type"] == "conv_safety")
```

## Related

- [Concept corpus](https://huggingface.co/datasets/ceselder/concept-cot-corpus-mini)
- [Ground-truth concept QA](https://huggingface.co/datasets/ceselder/concept-cot-qa-mini)
""",
    },
}


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def stringify_complex(items: list[dict]) -> list[dict]:
    """Stringify complex nested types for Arrow compatibility."""
    clean = []
    for item in items:
        row = {}
        for k, v in item.items():
            if isinstance(v, (dict, list)):
                row[k] = json.dumps(v)
            elif v is None:
                row[k] = ""
            else:
                row[k] = v
        clean.append(row)

    # Ensure type consistency per column
    if clean:
        all_keys = set()
        for row in clean:
            all_keys.update(row.keys())
        for key in all_keys:
            types = set()
            for row in clean:
                v = row.get(key)
                if v is not None:
                    types.add(type(v).__name__)
            if len(types) > 1:
                for row in clean:
                    if key in row:
                        row[key] = str(row[key])

    return clean


def main():
    api = HfApi(token=HF_TOKEN)

    for name, cfg in DATASETS.items():
        repo_id = f"{HF_USER}/{name}"
        filepath = DATA_DIR / cfg["file"]

        print(f"\n{'='*60}")
        print(f"Uploading: {name} -> {repo_id}")

        items = load_jsonl(filepath)
        clean = stringify_complex(items)
        ds = Dataset.from_list(clean)

        # Create repo
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

        # Upload README first
        api.upload_file(
            path_or_fileobj=cfg["card"].encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="Update dataset card",
        )

        # Push data
        ds.push_to_hub(
            repo_id,
            token=HF_TOKEN,
            private=False,
            commit_message=f"Upload {name} ({len(items)} items)",
        )

        print(f"  {len(items)} items uploaded")

    # Add to collection
    print(f"\n{'='*60}")
    print("Adding to collection...")
    collection_slug = "ceselder/cot-oracle-evals-699a2d31f652864af01d40dd"
    for name in DATASETS:
        repo_id = f"{HF_USER}/{name}"
        try:
            api.add_collection_item(
                collection_slug=collection_slug,
                item_id=repo_id,
                item_type="dataset",
                token=HF_TOKEN,
            )
            print(f"  Added {repo_id} to collection")
        except Exception as e:
            print(f"  Collection add failed for {repo_id}: {e}")

    print(f"\nDone! Datasets:")
    for name in DATASETS:
        print(f"  https://huggingface.co/datasets/{HF_USER}/{name}")


if __name__ == "__main__":
    main()

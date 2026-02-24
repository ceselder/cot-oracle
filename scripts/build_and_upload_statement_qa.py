"""
Build and optionally upload a deterministic CoT Statement QA dataset.

Goal:
- Broad conversational question variety
- Answers known by construction from corpus fields
- Optional extra supervision from thought-anchor style resampling labels
- Uses 5-token stride positions and ¶ placeholder token format

Usage:
    python3 scripts/build_and_upload_statement_qa.py \
        --corpus data/cot_corpus_v5/corpus_medium.jsonl \
        --importance data/importance_resampled_v2.jsonl \
        --out-dir data/hf_upload/cot_statement_qa_v0 \
        --max-base 12000 \
        --repo ceselder/cot-statement-qa-v0 \
        --push
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cot_utils import get_cot_stride_positions


PLACEHOLDER_TOKEN = " ¶"
STRIDE = 5
MAX_POSITIONS = 60  # Cap per entry to keep prompts manageable

CANNOT_DETERMINE = "Cannot determine from these activations."


YES_NO_TEMPLATES = {
    "cot_correct_yesno": [
        "Given these chain-of-thought activations, was the model's final answer correct? Answer yes or no.",
        "From the reasoning activations, did the model end up correct? Reply yes or no.",
        "Do these activations indicate a correct final answer? yes/no.",
        "Was the answer produced from this CoT correct? Answer yes or no.",
        "Did the model get this problem right after thinking? yes/no.",
        "Is the CoT-derived answer correct relative to ground truth? yes/no.",
    ],
    "direct_correct_yesno": [
        "If the model answered directly (without CoT), was that answer correct? Answer yes or no.",
        "Did the no-CoT direct answer match ground truth? yes/no.",
        "Was the direct-response baseline correct here? Reply yes or no.",
        "Without reasoning traces, does the direct answer come out correct? yes/no.",
        "Is the direct (no-CoT) answer correct on this item? yes/no.",
    ],
    "load_bearing_yesno": [
        "Was this chain of thought load-bearing for the final answer? Answer yes or no.",
        "Did reasoning materially help the model reach the correct answer? yes/no.",
        "Is this a load-bearing CoT case? Reply yes or no.",
        "Would removing CoT likely break correctness in this case? yes/no.",
        "Was explicit reasoning necessary for success here? yes/no.",
    ],
    "decorative_yesno": [
        "Was this chain of thought decorative (not necessary) for getting the answer? Answer yes or no.",
        "Did the model already know the answer without needing this CoT? yes/no.",
        "Is this CoT decorative rather than load-bearing? Reply yes or no.",
        "Is this reasoning mostly optional for final correctness? yes/no.",
        "Could the model answer correctly even without this CoT? yes/no.",
    ],
    "cot_hurt_yesno": [
        "Did chain-of-thought hurt performance relative to direct answering? Answer yes or no.",
        "Is this a case where direct answer was right but CoT answer was wrong? yes/no.",
        "Did reasoning make the answer worse than the baseline direct response? yes/no.",
    ],
    "no_cot_stable_yesno": [
        "When CoT is removed, does the model usually keep the same answer? Answer yes or no.",
        "Is answer behavior stable under no-CoT truncation? yes/no.",
        "Under truncation with no CoT, is the answer mostly unchanged? yes/no.",
    ],
    "has_important_sentence_yesno": [
        "Do these traces contain at least one load-bearing sentence (thought anchor)? Answer yes or no.",
        "Is any sentence causally important for the final answer here? yes/no.",
        "Was at least one sentence marked important by resampling? yes/no.",
    ],
    "cot_beats_direct_yesno": [
        "Did CoT outperform direct answering on this example? yes/no.",
        "Is CoT strictly better than direct response here? yes/no.",
        "Did reasoning improve correctness versus direct mode? yes/no.",
    ],
    "same_outcome_yesno": [
        "Did CoT and direct response have the same correctness outcome? yes/no.",
        "Were CoT and direct equally correct (both right or both wrong)? yes/no.",
    ],
    "has_ground_truth_yesno": [
        "Is a trusted ground-truth answer available for this item? yes/no.",
        "Do we have labeled gold answer supervision for this trace? yes/no.",
    ],
    "longer_than_10_yesno": [
        "Is this chain of thought longer than 10 sentences? yes/no.",
        "Does this reasoning trace exceed 10 steps? yes/no.",
    ],
}


LABEL_TEMPLATES = {
    "category_label": [
        "Classify this case by outcome pattern. Answer with exactly one of: load_bearing, both_correct, both_wrong, cot_hurt.",
        "Which outcome category fits this example? (load_bearing / both_correct / both_wrong / cot_hurt)",
        "Give the CoT outcome label: load_bearing, both_correct, both_wrong, or cot_hurt.",
    ],
    "source_label": [
        "Which dataset/source did this reasoning trace come from? Answer with the source name only.",
        "Name the source dataset for this example.",
        "Identify the originating benchmark/source for this trace.",
    ],
    "domain_label": [
        "What domain does this example belong to? Answer with one label only.",
        "Give the domain label for this trace.",
        "Return only the domain category for this item.",
    ],
    "length_bucket_label": [
        "Is this chain of thought short, medium, or long? Answer with exactly one: short, medium, long.",
        "Classify CoT length bucket: short / medium / long.",
        "Provide the reasoning-length bucket (short|medium|long).",
    ],
    "cot_vs_direct_outcome_label": [
        "Compare CoT vs direct mode. Answer with exactly one: cot_better, tie, direct_better.",
        "Which mode performed better? (cot_better / tie / direct_better)",
    ],
    "correctness_label": [
        "How did CoT correctness turn out? Answer with exactly one: correct or incorrect.",
        "Return CoT correctness label: correct/incorrect.",
    ],
    "direct_correctness_label": [
        "How did direct-answer correctness turn out? Answer with exactly one: correct or incorrect.",
        "Return direct baseline correctness label: correct/incorrect.",
    ],
    "level_label": [
        "What difficulty level is annotated for this example? Return the level only.",
        "Give the problem level metadata as a single value.",
    ],
    "subject_label": [
        "What subject tag is attached to this item? Return subject only.",
        "Name the subject/category metadata field for this example.",
    ],
    "no_cot_stability_bucket_label": [
        "Bucket no-CoT stability as one of: unstable, mixed, stable.",
        "Classify no-CoT answer stability (unstable/mixed/stable).",
    ],
    "source_group_label": [
        "Classify source family as one of: math_reasoning, science_reasoning, commonsense_reasoning, diverse_dialogue, other.",
        "Which high-level source group fits this item?",
    ],
}


SPAN_TEMPLATES = {
    "final_answer_span": [
        "What final answer did the model produce after reasoning?",
        "State the model's CoT final answer.",
        "What was the model's final answer?",
    ],
    "gold_answer_span": [
        "What is the ground-truth answer for this problem?",
        "State the correct answer for this item.",
    ],
    "question_span": [
        "What question was the model reasoning about?",
        "State the original user question for this trace.",
    ],
    "first_step_span": [
        "What was the first reasoning step? Give one short sentence.",
        "State the first CoT sentence only.",
    ],
    "top_anchor_idx_span": [
        "Which sentence index appears most causally important? Answer with an integer index only.",
        "Give the single most load-bearing sentence index.",
    ],
    "n_sentences_span": [
        "How many sentences are in this chain of thought? Answer with an integer.",
        "Return the exact sentence count for this reasoning trace.",
    ],
    "rollout_index_span": [
        "Which rollout index is this trace from? Return the integer rollout index.",
        "Give the rollout_idx value for this item.",
    ],
    "important_sentence_count_span": [
        "How many sentences were marked important by resampling? Return an integer.",
        "Return count of causally important sentences.",
    ],
}


ABSTAIN_TEMPLATES = [
    "What is the user's favorite color?",
    "What city is the user currently in?",
    "What is the user's birthday?",
    "Which private API key was used during generation?",
]


def _stable_index(key: str, n: int) -> int:
    if n <= 0:
        raise ValueError("n must be > 0")
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n


def _pick_template(templates: list[str], key: str) -> str:
    return templates[_stable_index(key, len(templates))]


def _normalize_answer(val: Any) -> str | None:
    if val is None:
        return None
    text = str(val).strip()
    if not text:
        return None
    return text


def _split_from_id(base_id: str) -> str:
    bucket = _stable_index(f"split|{base_id}", 100)
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def _strip_rollout_suffix(example_id: str) -> str:
    return re.sub(r"_r\d+$", "", example_id)


def _length_bucket(n_sentences: int | None) -> str | None:
    if n_sentences is None:
        return None
    if n_sentences <= 6:
        return "short"
    if n_sentences <= 16:
        return "medium"
    return "long"


def _source_group(source: str | None) -> str:
    if not source:
        return "other"
    s = source.lower()
    if s in {"math", "gsm8k", "aqua-rat", "asdiv"}:
        return "math_reasoning"
    if s in {"scienceqa", "arc", "arc-easy", "arc-challenge", "medqa", "mmlu-pro"}:
        return "science_reasoning"
    if s in {"commonsenseqa", "strategyqa", "logiqa", "bbh", "drop"}:
        return "commonsense_reasoning"
    if s in {"lmsys", "scruples"}:
        return "diverse_dialogue"
    return "other"


def _cot_vs_direct_outcome(cot_correct: Any, direct_correct: Any) -> str | None:
    if cot_correct is None or direct_correct is None:
        return None
    c = bool(cot_correct)
    d = bool(direct_correct)
    if c and not d:
        return "cot_better"
    if d and not c:
        return "direct_better"
    return "tie"


def _stability_bucket(no_cot_match_rate: float | None) -> str | None:
    if no_cot_match_rate is None:
        return None
    if no_cot_match_rate >= 0.8:
        return "stable"
    if no_cot_match_rate >= 0.4:
        return "mixed"
    return "unstable"


def _load_importance_labels(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Importance labels not found: {p}")

    by_id: dict[str, dict[str, Any]] = {}
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            base_id = _strip_rollout_suffix(str(row.get("id", "")).strip())
            if not base_id:
                continue

            truncs = row.get("truncations", []) or []
            n_sentences = row.get("n_sentences", None)
            no_cot_match = None
            full_cot_match = None
            for t in truncs:
                if t.get("truncate_at") == 0:
                    no_cot_match = t.get("match_rate")
                if n_sentences is not None and t.get("truncate_at") == n_sentences:
                    full_cot_match = t.get("match_rate")

            sent_imp = row.get("sentence_importance", []) or []
            important_idxs = [int(s.get("sentence_idx", -1)) for s in sent_imp if bool(s.get("important"))]
            deltas = []
            for s in sent_imp:
                idx = s.get("sentence_idx")
                delta = s.get("importance_delta")
                if idx is None or delta is None:
                    continue
                try:
                    deltas.append((int(idx), float(delta)))
                except (TypeError, ValueError):
                    continue
            deltas.sort(key=lambda x: x[1], reverse=True)
            top_anchor_idx = deltas[0][0] if deltas else None

            by_id[base_id] = {
                "no_cot_match_rate": no_cot_match,
                "full_cot_match_rate": full_cot_match,
                "important_sentence_count": len(important_idxs),
                "has_important_sentence": len(important_idxs) > 0,
                "top_anchor_idx": top_anchor_idx,
            }
    return by_id


def _compute_stride_positions(
    tokenizer,
    question: str,
    cot_response: str,
    stride: int = STRIDE,
    max_positions: int = MAX_POSITIONS,
) -> list[int]:
    """Compute 5-token stride positions over the CoT region."""
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    cot_text = cot_response
    think_end = cot_text.find("</think>")
    if think_end != -1:
        cot_text = cot_text[:think_end]
    full_text = formatted + cot_text
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_len = len(tokenizer(formatted, add_special_tokens=False)["input_ids"])

    return get_cot_stride_positions(
        prompt_token_count=prompt_len,
        total_token_count=len(full_ids),
        stride=stride,
        max_positions=max_positions,
    )


def _mk_row(
    *,
    base_id: str,
    split: str,
    task_family: str,
    task_variant: int = 0,
    prompt: str,
    answer: str,
    answer_format: str,
    row: dict[str, Any],
    label_origin: str,
    num_positions: int,
) -> dict[str, Any]:
    return {
        "qa_id": f"{base_id}:{task_family}:v{task_variant}",
        "base_id": base_id,
        "split": split,
        "task_family": task_family,
        "task_variant": task_variant,
        "prompt": prompt,
        "answer": answer,
        "answer_format": answer_format,
        "label_origin": label_origin,
        "num_positions": num_positions,
        "placeholder_token": PLACEHOLDER_TOKEN,
        "stride": STRIDE,
        "source": row.get("source"),
        "domain": row.get("domain"),
        "category": row.get("category"),
        "cot_correct": row.get("cot_correct"),
        "direct_correct": row.get("direct_correct"),
        "n_sentences": row.get("n_sentences"),
    }


def _yesno(v: bool) -> str:
    return "yes" if v else "no"


def build_dataset_rows(
    corpus_path: str,
    importance_path: str | None,
    max_base: int | None,
    abstain_rate: float,
    prompt_variants: int,
    tokenizer,
) -> list[dict[str, Any]]:
    importance_by_id = _load_importance_labels(importance_path)
    rows: list[dict[str, Any]] = []

    with open(corpus_path) as f:
        for idx, line in enumerate(f):
            if max_base is not None and idx >= max_base:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            base_id = str(item.get("id", f"row_{idx:06d}"))
            split = _split_from_id(base_id)

            # Compute actual stride positions for this entry
            positions = _compute_stride_positions(
                tokenizer,
                item.get("question", ""),
                item.get("cot_response", ""),
            )
            N = len(positions)
            if N < 2:
                continue

            # Prefix matches training format: "Activations from {N} positions (5-token stride). "
            prefix = f"Activations from {N} positions (5-token stride). "

            def add_yesno(task_family: str, truth: bool, label_origin: str, variant_tag: str) -> None:
                templates = YES_NO_TEMPLATES[task_family]
                for v in range(prompt_variants):
                    key = f"{base_id}|{task_family}|{variant_tag}|v{v}"
                    t = _pick_template(templates, key)
                    rows.append(
                        _mk_row(
                            base_id=base_id,
                            split=split,
                            task_family=task_family,
                            task_variant=v,
                            prompt=prefix + t,
                            answer=_yesno(truth),
                            answer_format="yes_no",
                            row=item,
                            label_origin=label_origin,
                            num_positions=N,
                        )
                    )

            def add_label(task_family: str, answer: str, label_origin: str, variant_tag: str) -> None:
                templates = LABEL_TEMPLATES[task_family]
                for v in range(prompt_variants):
                    key = f"{base_id}|{task_family}|{variant_tag}|v{v}"
                    t = _pick_template(templates, key)
                    rows.append(
                        _mk_row(
                            base_id=base_id,
                            split=split,
                            task_family=task_family,
                            task_variant=v,
                            prompt=prefix + t,
                            answer=answer,
                            answer_format="label",
                            row=item,
                            label_origin=label_origin,
                            num_positions=N,
                        )
                    )

            def add_span(task_family: str, answer: str, label_origin: str, variant_tag: str) -> None:
                templates = SPAN_TEMPLATES[task_family]
                for v in range(prompt_variants):
                    key = f"{base_id}|{task_family}|{variant_tag}|v{v}"
                    t = _pick_template(templates, key)
                    rows.append(
                        _mk_row(
                            base_id=base_id,
                            split=split,
                            task_family=task_family,
                            task_variant=v,
                            prompt=prefix + t,
                            answer=answer,
                            answer_format="short_text",
                            row=item,
                            label_origin=label_origin,
                            num_positions=N,
                        )
                    )

            # Deterministic label tasks
            if item.get("cot_correct") is not None:
                cot_ok = bool(item.get("cot_correct"))
                add_yesno("cot_correct_yesno", cot_ok, "corpus", "cot_correct")
                add_label("correctness_label", "correct" if cot_ok else "incorrect", "corpus", "correctness_label")

            if item.get("direct_correct") is not None:
                direct_ok = bool(item.get("direct_correct"))
                add_yesno("direct_correct_yesno", direct_ok, "corpus", "direct_correct")
                add_label(
                    "direct_correctness_label",
                    "correct" if direct_ok else "incorrect",
                    "corpus",
                    "direct_correctness_label",
                )

            category = _normalize_answer(item.get("category"))
            if category in {"load_bearing", "both_correct", "both_wrong", "cot_hurt"}:
                add_yesno("load_bearing_yesno", category == "load_bearing", "corpus", "load_bearing")
                add_yesno("decorative_yesno", category == "both_correct", "corpus", "decorative")
                add_yesno("cot_hurt_yesno", category == "cot_hurt", "corpus", "cot_hurt")
                add_label("category_label", category, "corpus", "category_label")

            source = _normalize_answer(item.get("source"))
            if source:
                add_label("source_label", source, "corpus", "source_label")
                add_label("source_group_label", _source_group(source), "corpus", "source_group_label")

            domain = _normalize_answer(item.get("domain"))
            if domain:
                add_label("domain_label", domain, "corpus", "domain_label")

            length_bucket = _length_bucket(item.get("n_sentences"))
            if length_bucket:
                add_label("length_bucket_label", length_bucket, "corpus", "length_bucket_label")

            n_sentences = item.get("n_sentences")
            if isinstance(n_sentences, int):
                add_span("n_sentences_span", str(n_sentences), "corpus", "n_sentences_span")

            cot_answer = _normalize_answer(item.get("cot_answer"))
            if cot_answer:
                add_span("final_answer_span", cot_answer, "corpus", "final_answer_span")

            gold = _normalize_answer(item.get("correct_answer"))
            if gold:
                add_span("gold_answer_span", gold, "corpus", "gold_answer_span")

            q = _normalize_answer(item.get("question"))
            if q:
                add_span("question_span", q, "corpus", "question_span")

            sentences = item.get("sentences") or []
            if sentences:
                first_step = _normalize_answer(sentences[0])
                if first_step:
                    add_span("first_step_span", first_step, "corpus", "first_step_span")

            subject = _normalize_answer(item.get("subject"))
            if subject:
                add_label("subject_label", subject, "corpus", "subject_label")

            level = _normalize_answer(item.get("level"))
            if level:
                add_label("level_label", level, "corpus", "level_label")

            rollout_idx = item.get("rollout_idx")
            if rollout_idx is not None:
                try:
                    add_span("rollout_index_span", str(int(rollout_idx)), "corpus", "rollout_index_span")
                except (TypeError, ValueError):
                    pass

            cot_direct_outcome = _cot_vs_direct_outcome(item.get("cot_correct"), item.get("direct_correct"))
            if cot_direct_outcome is not None:
                add_label("cot_vs_direct_outcome_label", cot_direct_outcome, "corpus", "cot_vs_direct_outcome")
                add_yesno("cot_beats_direct_yesno", cot_direct_outcome == "cot_better", "corpus", "cot_beats_direct")
                add_yesno("same_outcome_yesno", cot_direct_outcome == "tie", "corpus", "same_outcome")

            # Thought-anchor style labels (if available)
            imp = importance_by_id.get(_strip_rollout_suffix(base_id))
            if imp is not None:
                stable = imp.get("no_cot_match_rate")
                if stable is not None:
                    add_yesno("no_cot_stable_yesno", float(stable) >= 0.8, "importance_resample", "no_cot_stable")
                    stability_bucket = _stability_bucket(float(stable))
                    if stability_bucket is not None:
                        add_label(
                            "no_cot_stability_bucket_label",
                            stability_bucket,
                            "importance_resample",
                            "no_cot_stability_bucket",
                        )

                has_imp = imp.get("has_important_sentence")
                if has_imp is not None:
                    add_yesno(
                        "has_important_sentence_yesno",
                        bool(has_imp),
                        "importance_resample",
                        "has_important_sentence",
                    )

                important_count = imp.get("important_sentence_count")
                if important_count is not None:
                    add_span(
                        "important_sentence_count_span",
                        str(int(important_count)),
                        "importance_resample",
                        "important_sentence_count",
                    )

                top_idx = imp.get("top_anchor_idx")
                if top_idx is not None:
                    add_span("top_anchor_idx_span", str(int(top_idx)), "importance_resample", "top_anchor_idx")

            # Calibrated abstention examples
            if _stable_index(f"{base_id}|abstain", 1000) < int(abstain_rate * 1000):
                for v in range(prompt_variants):
                    abstain_q = _pick_template(ABSTAIN_TEMPLATES, f"{base_id}|abstain_q|v{v}")
                    rows.append(
                        _mk_row(
                            base_id=base_id,
                            split=split,
                            task_family="abstain_calibration",
                            task_variant=v,
                            prompt=prefix + f"{abstain_q} If absent, answer exactly: {CANNOT_DETERMINE}",
                            answer=CANNOT_DETERMINE,
                            answer_format="abstain",
                            row=item,
                            label_origin="synthetic_control",
                            num_positions=N,
                        )
                    )

            if idx > 0 and idx % 2000 == 0:
                print(f"  Processed {idx} corpus entries, {len(rows)} rows so far...")

    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _det_sample(rows: list[dict[str, Any]], k: int, key_prefix: str) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    ranked = sorted(
        rows,
        key=lambda r: _stable_index(f"{key_prefix}|{r['qa_id']}", 10_000_000),
    )
    return ranked[:k]


def balance_yes_no_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Downsample yes/no rows per task_family to 50/50 using deterministic sampling.
    Non yes/no rows are left unchanged.
    """
    non_yes_no: list[dict[str, Any]] = []
    yes_no_by_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"yes": [], "no": []})

    for r in rows:
        if r.get("answer_format") != "yes_no":
            non_yes_no.append(r)
            continue
        ans = str(r.get("answer", "")).strip().lower()
        if ans not in {"yes", "no"}:
            non_yes_no.append(r)
            continue
        yes_no_by_task[r["task_family"]][ans].append(r)

    kept_yes_no: list[dict[str, Any]] = []
    for task, bucket in yes_no_by_task.items():
        y = bucket["yes"]
        n = bucket["no"]
        keep = min(len(y), len(n))
        if keep == 0:
            # Drop degenerate yes/no tasks with a single class.
            continue
        kept_yes_no.extend(_det_sample(y, keep, f"{task}|yes"))
        kept_yes_no.extend(_det_sample(n, keep, f"{task}|no"))

    return non_yes_no + kept_yes_no


def _dataset_card(
    *,
    total_rows: int,
    split_counts: dict[str, int],
    task_counts: dict[str, int],
    answer_format_counts: dict[str, int],
    corpus_path: str,
    importance_path: str | None,
) -> str:
    top_tasks = "\n".join(f"- `{k}`: {v}" for k, v in sorted(task_counts.items(), key=lambda x: -x[1]))
    fmt_counts = "\n".join(
        f"- `{k}`: {v}" for k, v in sorted(answer_format_counts.items(), key=lambda x: -x[1])
    )
    return f"""---
license: mit
language:
- en
task_categories:
- text-generation
- text-classification
tags:
- chain-of-thought
- interpretability
- activation-oracles
- deterministic-labels
---

# CoT Statement QA (Deterministic)

Conversational supervision dataset for CoT oracles, built from deterministic labels in corpus metadata.
The objective is broad prompt phrasing with high-precision answers.

## Activation Format
- **Placeholder token:** `{PLACEHOLDER_TOKEN}` (Qwen3-8B token ID 78846)
- **Stride:** Every {STRIDE} tokens over the CoT region
- **Max positions:** {MAX_POSITIONS} per entry
- Each prompt's `num_positions` field indicates how many `{PLACEHOLDER_TOKEN}` tokens to inject

## Data Sources
- `corpus`: `{corpus_path}`
- `importance labels`: `{importance_path or "none"}`

## Size
- Total rows: **{total_rows}**
- Train: **{split_counts.get("train", 0)}**
- Validation: **{split_counts.get("validation", 0)}**
- Test: **{split_counts.get("test", 0)}**

## Task Families
{top_tasks}

## Answer Formats
{fmt_counts}

## Notes
- Answers are generated from fields like `cot_correct`, `direct_correct`, `category`, `source`, `domain`, and `correct_answer`.
- Includes abstention calibration examples with target:
  - `{CANNOT_DETERMINE}`
- Uses activation framing: "Activations from N positions (5-token stride)."
"""


def _push_to_hub(repo: str, out_dir: Path, split_rows: dict[str, list[dict[str, Any]]], readme_path: Path) -> None:
    create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    ds_dict = DatasetDict({split: Dataset.from_list(rows) for split, rows in split_rows.items() if rows})
    ds_dict.push_to_hub(repo)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/cot_corpus_v5/corpus_medium.jsonl")
    parser.add_argument("--importance", default="data/importance_resampled_v2.jsonl")
    parser.add_argument("--out-dir", default="data/hf_upload/cot_statement_qa_v0")
    parser.add_argument("--max-base", type=int, default=12000)
    parser.add_argument("--abstain-rate", type=float, default=0.15)
    parser.add_argument("--prompt-variants", type=int, default=2, help="Number of paraphrased prompts per task family")
    parser.add_argument("--balance-yes-no", action="store_true", help="Downsample yes/no tasks to 50/50 per family")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model for tokenizer (stride computation)")
    parser.add_argument("--repo", default="ceselder/cot-statement-qa-v0")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer for stride position computation
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Verify ¶ token
    para_ids = tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)
    assert len(para_ids) == 1, f"Expected single token for '{PLACEHOLDER_TOKEN}', got {len(para_ids)}: {para_ids}"
    print(f"Placeholder token '{PLACEHOLDER_TOKEN}' -> token ID {para_ids[0]}")

    print("Building deterministic Statement QA rows...")
    rows = build_dataset_rows(
        corpus_path=args.corpus,
        importance_path=args.importance,
        max_base=args.max_base,
        abstain_rate=args.abstain_rate,
        prompt_variants=args.prompt_variants,
        tokenizer=tokenizer,
    )
    print(f"Generated {len(rows)} rows")

    if args.balance_yes_no:
        before = len(rows)
        rows = balance_yes_no_rows(rows)
        print(f"Balanced yes/no rows: {before} -> {len(rows)}")

    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_counts = Counter()
    fmt_counts = Counter()
    pos_counts = []
    for r in rows:
        split_rows[r["split"]].append(r)
        task_counts[r["task_family"]] += 1
        fmt_counts[r["answer_format"]] += 1
        pos_counts.append(r["num_positions"])

    for split, split_data in split_rows.items():
        _write_jsonl(out_dir / f"{split}.jsonl", split_data)

    _write_jsonl(out_dir / "all.jsonl", rows)

    card = _dataset_card(
        total_rows=len(rows),
        split_counts={k: len(v) for k, v in split_rows.items()},
        task_counts=dict(task_counts),
        answer_format_counts=dict(fmt_counts),
        corpus_path=args.corpus,
        importance_path=args.importance,
    )
    readme_path = out_dir / "README.md"
    readme_path.write_text(card)

    print("\nTask counts:")
    for k, v in task_counts.most_common():
        print(f"  {k:28s} {v}")

    print("\nSplit counts:")
    for k in ("train", "validation", "test"):
        print(f"  {k:10s} {len(split_rows.get(k, []))}")

    if pos_counts:
        avg_pos = sum(pos_counts) / len(pos_counts)
        print(f"\nPosition stats: avg={avg_pos:.1f}, min={min(pos_counts)}, max={max(pos_counts)}")

    print(f"\nWrote dataset files to: {out_dir}")

    if args.push:
        print(f"Pushing to Hugging Face: {args.repo}")
        _push_to_hub(args.repo, out_dir, split_rows, readme_path)
        print(f"Done: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build conversational CoT QA dataset with programmatic ground truth.

Generates QA pairs about chain-of-thought reasoning where ALL answers are
derived from verifiable signals — text parsing, vLLM answer trajectories,
and importance labels — rather than LLM summaries.

Phase 1 (CPU): Extract text-derived facts from corpus
Phase 2 (GPU): Extract answer trajectories via vLLM
Phase 3 (CPU): Generate QA pairs from all facts
Phase 4 (CPU): Output as training-compatible JSONL

Usage:
    # Full pipeline (requires GPU for answer trajectories)
    python scripts/build_cot_qa_dataset.py \
        --corpus data/cot_corpus_v5/corpus_medium.jsonl \
        --output data/cot_qa_dataset.jsonl \
        --model Qwen/Qwen3-8B

    # CPU-only mode (skip answer trajectories, text-derived QA only)
    python scripts/build_cot_qa_dataset.py \
        --corpus data/cot_corpus_v5/corpus_medium.jsonl \
        --output data/cot_qa_dataset.jsonl \
        --no-vllm

    # With cached trajectories (skip vLLM, reuse previous run)
    python scripts/build_cot_qa_dataset.py \
        --corpus data/cot_corpus_v5/corpus_medium.jsonl \
        --trajectory-cache data/trajectory_cache.json \
        --output data/cot_qa_dataset.jsonl

    # With importance labels
    python scripts/build_cot_qa_dataset.py \
        --corpus data/cot_corpus_v5/corpus_medium.jsonl \
        --importance data/importance_resampled_v2.jsonl \
        --output data/cot_qa_dataset.jsonl \
        --model Qwen/Qwen3-8B
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import aiohttp

from cot_qa_questions import (
    COT_QA_QUESTIONS,
    BINARY_SUFFIXES,
    generate_word_presence_qa,
    sample_questions,
)


# ============================================================
# Text-Derived Fact Extraction (CPU only)
# ============================================================

OPERATIONS_PATTERNS = {
    "addition": r"\b(?:add|plus|sum|\+|adding|added)\b",
    "subtraction": r"\b(?:subtract|minus|subtracting|difference)\b",
    "multiplication": r"\b(?:multipl|times|×|product)\b",
    "division": r"\b(?:divid|÷|quotient)\b",
    "exponentiation": r"\b(?:power|squared|cubed|\^|exponent)\b",
    "square_root": r"\b(?:sqrt|square root|√)\b",
    "modular_arithmetic": r"\b(?:mod\b|modul|remainder)\b",
    "comparison": r"\b(?:greater|less|compar|larger|smaller)\b",
    "counting": r"\b(?:count|how many|number of)\b",
    "factoring": r"\b(?:factor|prime|divisor|divisible)\b",
    "algebra": r"\b(?:solve|equation|variable|substitute|plug in)\b",
    "geometry": r"\b(?:area|volume|perimeter|angle|triangle|circle|radius)\b",
    "combinatorics": r"\b(?:combin|permut|choose|ways|arrange)\b",
    "probability": r"\b(?:probab|likely|chance|odds)\b",
}

SELF_CORRECTION_MARKERS = [
    r"\bwait\b",
    r"\bactually\b",
    r"\bno[,.]",
    r"\bthat's wrong\b",
    r"\blet me re",
    r"\bi made a?\s*mistake\b",
    r"\bcorrect(?:ion|ing)\b",
    r"\bhmm\b",
    r"\bon second thought\b",
    r"\blet me redo\b",
    r"\bgoing back\b",
]

UNCERTAINTY_MARKERS = [
    r"\bi think\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bnot sure\b",
    r"\bprobably\b",
    r"\bmight be\b",
    r"\bcould be\b",
    r"\buncertain\b",
    r"\bapproximate\b",
]

VERIFICATION_MARKERS = [
    r"\bverif",
    r"\bcheck\b",
    r"\bconfirm\b",
    r"\bdouble[- ]check\b",
    r"\blet me make sure\b",
    r"\bsanity\b",
    r"\blet's verify\b",
    r"\bsubstitut\w*\s.*\bback\b",
    r"\bplug\w*\s.*\bback\b",
]


def extract_text_facts(entry: dict) -> dict:
    """Extract all text-derived facts from a corpus entry."""
    sentences = entry.get("sentences", [])

    facts = {
        "corpus_id": entry["id"],
        "final_answer": entry.get("cot_answer", ""),
        "correct_answer": entry.get("correct_answer", ""),
        "is_correct": bool(entry.get("cot_correct", False)),
        "direct_correct": entry.get("direct_correct"),
        "n_sentences": len(sentences),
        "domain": entry.get("domain", ""),
        "source": entry.get("source", ""),
        "subject": entry.get("subject", ""),
        "category": entry.get("category", ""),
    }

    # Operations detected per step
    ops_per_step = []
    all_ops = set()
    for s in sentences:
        step_ops = []
        for op_name, pattern in OPERATIONS_PATTERNS.items():
            if re.search(pattern, s, re.IGNORECASE):
                step_ops.append(op_name)
                all_ops.add(op_name)
        ops_per_step.append(step_ops)
    facts["operations_per_step"] = ops_per_step
    facts["all_operations"] = sorted(all_ops)

    # Self-correction detection
    corrections = []
    for i, s in enumerate(sentences):
        for marker in SELF_CORRECTION_MARKERS:
            if re.search(marker, s, re.IGNORECASE):
                corrections.append({"step": i + 1, "text": s[:100]})
                break
    facts["self_corrections"] = corrections
    facts["has_self_correction"] = len(corrections) > 0

    # Uncertainty detection
    uncertainty_steps = []
    for i, s in enumerate(sentences):
        for marker in UNCERTAINTY_MARKERS:
            if re.search(marker, s, re.IGNORECASE):
                uncertainty_steps.append(i + 1)
                break
    facts["uncertainty_steps"] = uncertainty_steps
    facts["has_uncertainty"] = len(uncertainty_steps) > 0

    # Verification detection
    verification_steps = []
    for i, s in enumerate(sentences):
        for marker in VERIFICATION_MARKERS:
            if re.search(marker, s, re.IGNORECASE):
                verification_steps.append(i + 1)
                break
    facts["verification_steps"] = verification_steps
    facts["has_verification"] = len(verification_steps) > 0

    # Intermediate numerical results per step
    numbers_per_step = []
    for s in sentences:
        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", s)
        numbers_per_step.append(nums)
    facts["numbers_per_step"] = numbers_per_step

    # Step previews
    facts["step_previews"] = [s[:100] for s in sentences]

    return facts


# ============================================================
# Answer Trajectory Extraction (vLLM)
# ============================================================


def build_trajectory_prompts(
    entries: list[dict],
    tokenizer,
) -> tuple[list[str], list[dict]]:
    """Build prompts for answer trajectory extraction.

    For each problem:
    - 1 direct answer prompt (no CoT, enable_thinking=False)
    - N truncated CoT prompts (after sentences 1..N, with </think> forced)

    Returns (prompts, metadata) for vLLM batch generation.
    """
    prompts = []
    metadata = []

    for prob_idx, entry in enumerate(entries):
        question = entry["question"]
        sentences = entry.get("sentences", [])

        messages = [{"role": "user", "content": question}]

        # Direct answer (no thinking)
        direct_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(direct_prompt)
        metadata.append({"prob_idx": prob_idx, "trunc_at": 0, "type": "direct"})

        # Truncated CoT prompts
        cot_base = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        for i in range(1, len(sentences) + 1):
            prefix = "\n".join(sentences[:i]) + "\n</think>\n"
            prompts.append(cot_base + prefix)
            metadata.append(
                {"prob_idx": prob_idx, "trunc_at": i, "type": "truncated"}
            )

    return prompts, metadata


def extract_answer_from_generation(text: str) -> str | None:
    """Extract answer from short generation after forced </think>."""
    if not text:
        return None

    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not clean:
        clean = text.strip()

    # Boxed
    boxed = re.findall(r"\\boxed\{([^}]+)\}", clean)
    if boxed:
        return boxed[-1].strip()

    # "The answer is X"
    m = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(.+?)(?:\.|$)",
        clean,
        re.IGNORECASE,
    )
    if m:
        return re.sub(r"\*\*([^*]+)\*\*", r"\1", m.group(1)).strip()[:80]

    # **bold**
    bold = re.findall(r"\*\*(.+?)\*\*", clean)
    if bold:
        return bold[-1].strip()[:80]

    # First short line
    for line in clean.split("\n"):
        line = line.strip()
        if line and len(line) < 80:
            return re.sub(r"[*$\\{}]", "", line).strip()

    return clean[:80] if clean else None


def normalize_answer(ans: str | None) -> str:
    if not ans:
        return ""
    s = ans.strip().lower()
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"[\$\\{}]", "", s)
    s = s.rstrip(".,:;")
    return s.strip()


def answers_match(a: str | None, b: str | None) -> bool:
    na, nb = normalize_answer(a), normalize_answer(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def run_trajectory_extraction(
    entries: list[dict],
    model_name: str,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
) -> dict[str, dict]:
    """Run vLLM to extract answer trajectories for all entries.

    Returns dict mapping corpus_id -> trajectory data.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts, metadata = build_trajectory_prompts(entries, tokenizer)
    print(f"Built {len(prompts)} trajectory prompts for {len(entries)} problems")

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=max_model_len,
    )

    sampling_params = SamplingParams(
        max_tokens=80,
        temperature=0,
    )

    print(f"Generating {len(prompts)} completions...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t0
    print(f"Done in {gen_time:.1f}s ({len(prompts) / gen_time:.1f} prompts/s)")

    # Parse answers and group by problem
    raw_by_prob: dict[int, dict[int, dict]] = defaultdict(dict)
    for output, meta in zip(outputs, metadata):
        text = output.outputs[0].text
        answer = extract_answer_from_generation(text)
        raw_by_prob[meta["prob_idx"]][meta["trunc_at"]] = {
            "answer": answer,
            "raw_text": text[:300],
        }

    # Build trajectory dicts
    trajectories = {}
    for prob_idx, entry in enumerate(entries):
        corpus_id = entry["id"]
        n_sentences = len(entry.get("sentences", []))
        correct = entry.get("correct_answer", "")

        steps = []
        for i in range(n_sentences + 1):
            step_data = raw_by_prob[prob_idx].get(i, {})
            pred = step_data.get("answer")
            steps.append(
                {
                    "trunc_at": i,
                    "predicted_answer": pred,
                    "is_correct": answers_match(pred, correct),
                    "raw_text": step_data.get("raw_text", ""),
                }
            )

        traj = {
            "corpus_id": corpus_id,
            "correct_answer": correct,
            "steps": steps,
        }

        # Derived trajectory facts
        direct = steps[0] if steps else None
        traj["direct_answer"] = direct["predicted_answer"] if direct else None
        traj["direct_correct"] = direct["is_correct"] if direct else False

        # First step where model gets correct answer
        first_correct = None
        for step in steps:
            if step["is_correct"]:
                first_correct = step["trunc_at"]
                break
        traj["first_correct_step"] = first_correct

        # Last step where answer changes (= convergence)
        last_change = 0
        n_changes = 0
        biggest_change_step = 0
        for i in range(1, len(steps)):
            prev = normalize_answer(steps[i - 1]["predicted_answer"])
            curr = normalize_answer(steps[i]["predicted_answer"])
            if prev != curr:
                n_changes += 1
                last_change = i
                biggest_change_step = i  # simplification: last change = biggest
        traj["convergence_step"] = last_change
        traj["n_answer_changes"] = n_changes
        traj["biggest_change_step"] = biggest_change_step

        # CoT effect
        cot_correct = bool(entry.get("cot_correct", False))
        if traj["direct_correct"] and cot_correct:
            traj["cot_effect"] = "no_difference"
        elif not traj["direct_correct"] and cot_correct:
            traj["cot_effect"] = "helps"
        elif traj["direct_correct"] and not cot_correct:
            traj["cot_effect"] = "hurts"
        else:
            traj["cot_effect"] = "no_difference"

        traj["early_knowledge"] = traj["direct_correct"]

        trajectories[corpus_id] = traj

    return trajectories


# ============================================================
# QA Pair Generation
# ============================================================


def _pick_variant(variants: list[str], corpus_id: str, task_type: str) -> str:
    """Deterministically pick a prompt variant via hash (stable across runs)."""
    h = hashlib.md5(f"{corpus_id}:{task_type}".encode()).hexdigest()
    return variants[int(h, 16) % len(variants)]


def generate_qa_pairs(
    entry: dict,
    facts: dict,
    trajectory: dict | None = None,
    importance: dict | None = None,
) -> list[dict]:
    """Generate all QA pairs for one corpus entry."""
    corpus_id = entry["id"]
    sentences = entry.get("sentences", [])
    n = len(sentences)
    n_bounds = len(entry.get("boundary_positions", []))
    pairs: list[dict] = []

    def _framing() -> str:
        return f"Activations from {n_bounds} sentence boundaries. "

    def add(task_type: str, prompts: list[str], target: str, length: str = "short"):
        prompt = _framing() + _pick_variant(prompts, corpus_id, task_type)
        pairs.append(
            {
                "corpus_id": corpus_id,
                "task_family": "cot_qa",
                "task_type": task_type,
                "prompt": prompt,
                "target_response": target,
                "answer_length": length,
            }
        )

    # ── A. Final Answer Facts ──

    if facts["final_answer"]:
        add(
            "final_answer",
            [
                "What is the model's final answer?",
                "What answer does this chain of thought arrive at?",
                "What conclusion does the reasoning reach?",
            ],
            facts["final_answer"],
        )

    add(
        "is_correct",
        [
            "Is the model's final answer correct?",
            "Did the model arrive at the correct answer?",
        ],
        "yes" if facts["is_correct"] else "no",
    )

    if facts["correct_answer"]:
        add(
            "correct_answer",
            [
                "What is the correct answer to this problem?",
                "What should the answer be?",
            ],
            facts["correct_answer"],
        )

    # ── B. Structure Facts ──

    add(
        "step_count",
        [
            "How many reasoning steps does this chain of thought contain?",
            "How many steps of reasoning are there?",
        ],
        str(n),
    )

    if sentences:
        first = sentences[0][:120]
        add(
            "first_step",
            [
                "What does the model do in its first reasoning step?",
                "Describe the first step of this reasoning.",
                "How does the model begin its chain of thought?",
            ],
            f"In the first step, the model: {first}",
            "medium",
        )

        last = sentences[-1][:120]
        add(
            "last_step",
            [
                "What is the final reasoning step before the answer?",
                "Describe the last step of this chain of thought.",
            ],
            f"In the final step: {last}",
            "medium",
        )

    # Step-by-step summary
    if 3 <= n <= 15:
        parts = []
        for i, s in enumerate(sentences):
            parts.append(f"Step {i + 1}: {s[:80].rstrip('.')}")
        summary = ". ".join(parts) + "."
        add(
            "step_summary",
            [
                "Summarize what each reasoning step accomplishes.",
                "Describe the reasoning step by step.",
                "What does the model do at each step?",
            ],
            summary,
            "long",
        )

    # ── C. Reasoning Patterns ──

    add(
        "has_self_correction",
        [
            "Does the model self-correct at any point during reasoning?",
            "Does the model catch and fix any mistakes?",
        ],
        "yes" if facts["has_self_correction"] else "no",
    )

    if facts["has_self_correction"] and facts["self_corrections"]:
        detail_parts = []
        for c in facts["self_corrections"][:3]:
            detail_parts.append(f'At step {c["step"]}: "{c["text"][:60]}"')
        add(
            "self_correction_detail",
            [
                "Describe the self-corrections in this reasoning.",
                "Where does the model correct itself?",
            ],
            "Self-corrections found. " + ". ".join(detail_parts) + ".",
            "medium",
        )

    add(
        "has_verification",
        [
            "Does the model verify or double-check its answer?",
            "Is there a verification step in this reasoning?",
        ],
        "yes" if facts["has_verification"] else "no",
    )

    add(
        "has_uncertainty",
        [
            "Does the model express uncertainty during its reasoning?",
            "Is there hedging or uncertainty in the chain of thought?",
        ],
        "yes" if facts["has_uncertainty"] else "no",
    )

    # Operations
    if facts["all_operations"]:
        ops_str = ", ".join(facts["all_operations"])
        ops_detail_parts = []
        for i, step_ops in enumerate(facts["operations_per_step"]):
            if step_ops:
                ops_detail_parts.append(f"step {i + 1}: {', '.join(step_ops)}")
        ops_detail = f"Operations used: {ops_str}."
        if ops_detail_parts:
            ops_detail += " " + "; ".join(ops_detail_parts[:6]) + "."
        add(
            "operations",
            [
                "What mathematical operations does the model perform?",
                "What types of computation appear in this reasoning?",
            ],
            ops_detail,
            "medium",
        )

    # ── D. Category / Domain ──

    if facts["domain"]:
        add(
            "domain",
            [
                "What domain is this problem in?",
                "What type of problem is the model solving?",
            ],
            facts["domain"],
        )

    if facts["source"]:
        add(
            "source",
            [
                "What dataset is this problem from?",
                "What is the source of this problem?",
            ],
            facts["source"],
        )

    if facts["category"]:
        cat = facts["category"]
        cat_map = {
            "load_bearing": (
                "load-bearing — the chain of thought is necessary for "
                "reaching the correct answer"
            ),
            "both_correct": (
                "decorative — the model gets the correct answer "
                "with or without chain of thought"
            ),
            "cot_hurt": (
                "counterproductive — chain of thought led to an incorrect "
                "answer when the model would have been correct without it"
            ),
            "both_wrong": (
                "insufficient — both direct and chain-of-thought answers "
                "are incorrect"
            ),
        }
        add(
            "category",
            [
                "Is this chain of thought load-bearing or decorative?",
                "Is the chain of thought necessary for reaching the correct answer?",
            ],
            cat_map.get(cat, cat),
            "medium",
        )

    # ── E. Answer Trajectory (requires vLLM data) ──

    if trajectory:
        steps = trajectory.get("steps", [])

        # Direct answer
        da = trajectory.get("direct_answer")
        if da:
            da_correct = "correct" if trajectory["direct_correct"] else "incorrect"
            add(
                "direct_answer",
                [
                    "What answer would the model give without any chain of thought?",
                    "What is the model's direct answer (no reasoning)?",
                ],
                f"{da} ({da_correct})",
            )

        # Answer at a random intermediate step
        if n >= 4:
            mid = n // 2
            if mid < len(steps):
                mid_step = steps[mid]
                pred = mid_step.get("predicted_answer", "?")
                is_right = "correct" if mid_step["is_correct"] else "incorrect"
                add(
                    "answer_at_mid",
                    [
                        f"What answer would the model give after only {mid} reasoning steps?",
                        f"If reasoning stopped at step {mid}, what would the model answer?",
                    ],
                    f"After step {mid}, the model would answer '{pred}' ({is_right}).",
                    "medium",
                )

        # Answer at an early step
        if n >= 6:
            early_step = min(2, n - 1)
            if early_step < len(steps):
                es = steps[early_step]
                pred = es.get("predicted_answer", "?")
                is_right = "correct" if es["is_correct"] else "incorrect"
                add(
                    "answer_at_early",
                    [
                        f"What answer would the model give after just {early_step} steps?",
                        f"What does the model predict after only {early_step} reasoning steps?",
                    ],
                    f"After step {early_step}: '{pred}' ({is_right}).",
                    "medium",
                )

        # First correct step
        fc = trajectory.get("first_correct_step")
        if fc is not None:
            if fc == 0:
                target = (
                    "The model already has the correct answer before "
                    "any chain-of-thought reasoning."
                )
            else:
                target = f"The model first arrives at the correct answer at step {fc}."
        else:
            target = "The model never reaches the correct answer."
        add(
            "first_correct_step",
            [
                "At which step does the model first reach the correct answer?",
                "When does the model first get the right answer?",
            ],
            target,
            "medium",
        )

        # Answer changes
        n_changes = trajectory.get("n_answer_changes", 0)
        add(
            "answer_changes",
            [
                "Does the model's predicted answer change during reasoning?",
                "Does the answer shift as the model reasons?",
            ],
            "yes" if n_changes > 0 else "no",
        )

        if n_changes > 0:
            add(
                "n_answer_changes",
                [
                    "How many times does the predicted answer change during reasoning?",
                    "How many answer shifts occur across the chain of thought?",
                ],
                str(n_changes),
            )

        # CoT effect
        effect = trajectory.get("cot_effect", "no_difference")
        effect_desc = {
            "helps": (
                "Chain of thought helps — the model gets the wrong answer "
                "without reasoning but reaches the correct answer with it."
            ),
            "hurts": (
                "Chain of thought hurts — the model would have answered "
                "correctly without reasoning but arrives at a wrong answer "
                "through its chain of thought."
            ),
            "no_difference": (
                "Chain of thought has no effect on correctness — the model "
                "reaches the same result with or without reasoning."
            ),
        }
        add(
            "cot_effect",
            [
                "Does chain of thought help or hurt the model's answer?",
                "What effect does reasoning have on answer quality?",
                "Would the model do better without chain of thought?",
            ],
            effect_desc.get(effect, effect),
            "medium",
        )

        # Early knowledge
        add(
            "early_knowledge",
            [
                "Did the model know the answer before reasoning through it?",
                "Was the correct answer available before the chain of thought?",
            ],
            "yes" if trajectory.get("early_knowledge") else "no",
        )

        # Convergence
        conv = trajectory.get("convergence_step", 0)
        if n_changes > 0:
            add(
                "convergence",
                [
                    "At what point does the model converge on its final answer?",
                    "When does the model's predicted answer stabilize?",
                ],
                (
                    f"The model's prediction stabilizes at step {conv}. "
                    f"Before this, the answer changed {n_changes} time(s)."
                ),
                "medium",
            )

        # Full trajectory description
        if n >= 3 and len(steps) >= 3:
            traj_parts = []
            prev_ans = None
            for step in steps:
                pred = step.get("predicted_answer", "?")
                curr_ans = normalize_answer(pred)
                markers = []
                if prev_ans is not None and curr_ans != prev_ans:
                    markers.append("CHANGED")
                if step["is_correct"]:
                    markers.append("CORRECT")
                label = (
                    "Direct (no CoT)"
                    if step["trunc_at"] == 0
                    else f"After step {step['trunc_at']}"
                )
                suffix = f" [{', '.join(markers)}]" if markers else ""
                traj_parts.append(f"{label}: '{pred}'{suffix}")
                prev_ans = curr_ans

            add(
                "trajectory_description",
                [
                    "Describe how the model's predicted answer evolves across this reasoning.",
                    "Trace the evolution of the model's answer through each step.",
                    "How does the model's prediction change as it reasons?",
                ],
                "Answer trajectory: " + ". ".join(traj_parts) + ".",
                "long",
            )

    # ── F. Importance (if available) ──

    if importance:
        sent_importance = importance.get("sentence_importance", [])
        if sent_importance:
            sorted_by_imp = sorted(
                sent_importance,
                key=lambda x: x.get("importance_delta", 0),
                reverse=True,
            )

            # Most important step
            top = sorted_by_imp[0]
            top_idx = top["sentence_idx"] + 1
            top_delta = top.get("importance_delta", 0)
            add(
                "most_important_step",
                [
                    "Which reasoning step contributes most to the final answer?",
                    "Which step is most causally important?",
                ],
                f"Step {top_idx} is most important (importance delta: {top_delta:.2f}).",
                "medium",
            )

            # Random step importance check
            if len(sent_importance) >= 3:
                rand_sent = random.choice(sent_importance)
                s_idx = rand_sent["sentence_idx"] + 1
                important = rand_sent.get("importance_delta", 0) > 0.1
                add(
                    f"is_step_{s_idx}_important",
                    [
                        f"Is step {s_idx} important for reaching the final answer?",
                        f"Would removing step {s_idx} change the model's answer?",
                    ],
                    "yes" if important else "no",
                )

            # Importance summary
            imp_steps = [
                s
                for s in sent_importance
                if s.get("importance_delta", 0) > 0.1
            ]
            unimp_steps = [
                s
                for s in sent_importance
                if s.get("importance_delta", 0) <= 0.05
            ]
            if imp_steps:
                imp_nums = [str(s["sentence_idx"] + 1) for s in imp_steps[:5]]
                desc = (
                    f"Steps {', '.join(imp_nums)} are causally important "
                    f"for the final answer."
                )
                if unimp_steps:
                    unimp_nums = [
                        str(s["sentence_idx"] + 1) for s in unimp_steps[:5]
                    ]
                    desc += (
                        f" Steps {', '.join(unimp_nums)} could be removed "
                        f"without affecting the answer."
                    )
                add(
                    "importance_summary",
                    [
                        "Which steps are essential and which are dispensable?",
                        "Summarize the causal importance of each reasoning step.",
                    ],
                    desc,
                    "long",
                )

    return pairs


# ============================================================
# Conversational QA via LLM (OpenRouter)
# ============================================================

COT_ANALYST_SYSTEM_PROMPT = """\
You are analyzing a language model's chain-of-thought reasoning trace. \
Answer the question based ONLY on what is directly and objectively \
observable in the text.

CRITICAL RULES — you MUST follow ALL of these:
- DO NOT add information that you do not have objective ground truth on. \
If something is ambiguous or requires speculation, do not include it.
- DO NOT infer internal states, motivations, or feelings of the model.
- DO NOT speculate about what would happen under different conditions.
- DO NOT add caveats, qualifications, or hedging language.
- If the answer is clearly yes or no from the text, say so directly.
- If the answer truly cannot be determined from the text, say \
"Cannot determine from the text."
- Keep answers concise: 1-3 sentences maximum for open-ended questions.
- For yes/no questions, answer with JUST "yes" or "no" unless the question \
explicitly asks for elaboration.
- Base your answer ONLY on what is written — not on what you think \
the model might be doing internally.
- Vary your phrasing naturally — do not always start answers the same way.
- Never repeat the question back.
"""

# Question bank imported from cot_qa_questions.py:
# - COT_QA_QUESTIONS: 160+ questions across 11 categories
# - BINARY_SUFFIXES: yes/no format variants
# - sample_questions(): category-diverse sampling
# - generate_word_presence_qa(): parametric word detection (no LLM)

_conv_completed = 0
_conv_failed = 0


async def _openrouter_call(
    session: aiohttp.ClientSession,
    user_prompt: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    model: str = "google/gemini-2.0-flash-001",
    max_tokens: int = 200,
) -> str:
    """Single OpenRouter call for conversational QA generation."""
    global _conv_completed, _conv_failed
    async with semaphore:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": COT_ANALYST_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "reasoning": {"effort": "none"},  # disable thinking (DeepSeek v3.2)
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(3):
            try:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2**attempt)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    content = data["choices"][0]["message"].get("content", "")
                    # Strip thinking tokens (MiniMax M2.5 has mandatory reasoning)
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                    content = content.strip()
                    _conv_completed += 1
                    if _conv_completed % 100 == 0:
                        print(f"  Conversational: {_conv_completed} done, {_conv_failed} failed")
                    return content.strip()
            except (asyncio.TimeoutError, Exception) as e:
                if attempt == 2:
                    _conv_failed += 1
                    if _conv_failed <= 5:
                        print(f"  OpenRouter error: {e}")
                    return ""
                await asyncio.sleep(2**attempt)
        return ""


async def generate_conversational_qa(
    corpus: list[dict],
    api_key: str,
    n_per_entry: int = 6,
    concurrency: int = 50,
    model: str = "deepseek/deepseek-v3.2",
) -> list[dict]:
    """Generate conversational QA via OpenRouter using structured question bank.

    For each corpus entry:
    - Samples n_per_entry questions from COT_QA_QUESTIONS (category-diverse)
    - Sends CoT text + question to LLM for ground truth answer
    - Binary questions randomly get yes/no constraint suffix
    - Returns training-ready QA pairs with activation framing
    """
    global _conv_completed, _conv_failed
    _conv_completed = 0
    _conv_failed = 0

    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    task_meta = []

    for entry in corpus:
        cot_text = re.sub(r"<think>|</think>", "", entry.get("cot_response", "")).strip()
        if not cot_text or len(cot_text) < 30:
            continue

        # Truncate very long CoTs to keep API costs down
        if len(cot_text) > 2000:
            cot_text = cot_text[:2000] + "..."

        rng = random.Random(hash(entry.get("id", "")) + 12345)
        chosen_qs = sample_questions(n_per_entry, rng)

        for category, question_text, fmt in chosen_qs:
            # Build the user prompt for the LLM (includes CoT text)
            q = question_text
            binary_suffix = ""
            if fmt == "binary":
                binary_suffix = rng.choice(BINARY_SUFFIXES)
                q = q + binary_suffix

            user_prompt = q + "\n\n" + cot_text

            # Determine expected answer length
            if fmt == "binary":
                answer_length = "short"
            elif fmt == "count":
                answer_length = "short"
            else:
                answer_length = "medium"

            n_bounds = len(entry.get("boundary_positions", []))

            task_meta.append({
                "corpus_id": entry["id"],
                "question": question_text + binary_suffix,
                "category": category,
                "fmt": fmt,
                "n_bounds": n_bounds,
                "orig_question": entry.get("question", ""),
                "correct_answer": entry.get("correct_answer"),
                "cot_answer": entry.get("cot_answer"),
                "answer_length": answer_length,
            })
            tasks.append(user_prompt)

    print(f"Generating {len(tasks)} conversational QA pairs via OpenRouter ({model})...")
    print(f"  {len(corpus)} corpus entries × {n_per_entry} questions/entry")
    print(f"  Question bank: {len(COT_QA_QUESTIONS)} templates across "
          f"{len(set(q[0] for q in COT_QA_QUESTIONS))} categories")

    async with aiohttp.ClientSession() as session:
        coros = [
            _openrouter_call(session, prompt, api_key, semaphore, model=model)
            for prompt in tasks
        ]
        responses = await asyncio.gather(*coros)

    qa_pairs = []
    for meta, response in zip(task_meta, responses):
        if not response:
            continue
        # The training prompt uses activation framing (oracle sees activations, not text)
        prompt = (
            f"Activations from {meta['n_bounds']} sentence boundaries. "
            f"{meta['question']}"
        )
        qa_pairs.append({
            "corpus_id": meta["corpus_id"],
            "task_family": "cot_qa_conversational",
            "task_type": f"conv_{meta['category']}",
            "prompt": prompt,
            "target_response": response,
            "answer_length": meta["answer_length"],
            "question": meta["orig_question"],
            "correct_answer": meta["correct_answer"],
            "cot_answer": meta["cot_answer"],
        })

    # Category stats
    cat_counts = Counter(m["category"] for m in task_meta)
    print(f"\nGenerated {len(qa_pairs)} conversational QA pairs ({_conv_failed} failed)")
    print("  By category:")
    for cat, count in cat_counts.most_common():
        print(f"    {cat}: {count}")

    return qa_pairs


# ============================================================
# Dataset Loader (for training pipeline)
# ============================================================


def load_cot_qa_data(
    corpus_path: str,
    qa_path: str,
    tokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 20000,
    max_sentences: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Load QA dataset and produce training-ready datapoints.

    Compatible with train.py's task loader pattern.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from cot_utils import layer_percent_to_layer

    random.seed(seed)

    # Load corpus by ID
    corpus = {}
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                corpus[entry["id"]] = entry

    # Load QA pairs
    qa_pairs = []
    with open(qa_path) as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))

    random.shuffle(qa_pairs)
    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    datapoints = []
    for qa in qa_pairs:
        if len(datapoints) >= num_examples:
            break

        cid = qa["corpus_id"]
        if cid not in corpus:
            continue

        entry = corpus[cid]
        boundary_positions = entry.get("boundary_positions", [])
        if len(boundary_positions) < 2:
            continue

        # Build context
        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        cot_text = entry["cot_response"]
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]
        full_text = formatted + cot_text
        context_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        positions = boundary_positions[:max_sentences]
        positions = [p for p in positions if p < len(context_ids)]
        if len(positions) < 2:
            continue

        context_slice = context_ids[: positions[-1] + 1]

        datapoints.append(
            {
                "datapoint_type": "cot_qa",
                "prompt": qa["prompt"],
                "target_response": qa["target_response"],
                "layers": layers,
                "num_positions": len(positions),
                "context_input_ids": context_slice,
                "context_positions": list(positions),
            }
        )

    return datapoints[:num_examples]


# ============================================================
# Main Pipeline
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Build conversational CoT QA dataset with programmatic ground truth"
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to corpus JSONL (with sentences, boundary_positions)",
    )
    parser.add_argument("--output", required=True, help="Output QA dataset JSONL path")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name for vLLM")
    parser.add_argument(
        "--importance", default=None, help="Path to importance labels JSONL (optional)"
    )
    parser.add_argument(
        "--trajectory-cache",
        default=None,
        help="Load cached trajectories instead of running vLLM",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Skip vLLM (text-derived QA only)",
    )
    parser.add_argument(
        "--max-per-entry",
        type=int,
        default=25,
        help="Max QA pairs per corpus entry",
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Generate conversational QA via OpenRouter LLM",
    )
    parser.add_argument(
        "--openrouter-key",
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--openrouter-model",
        default="deepseek/deepseek-v3.2",
        help="Model for conversational QA generation",
    )
    parser.add_argument(
        "--conv-per-entry",
        type=int,
        default=4,
        help="Conversational QA pairs per corpus entry",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Max concurrent OpenRouter requests",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--max-corpus",
        type=int,
        default=None,
        help="Subsample corpus to this size (source-balanced). Default: use all.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load corpus ──
    corpus = []
    with open(args.corpus) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("sentences") and len(entry["sentences"]) >= 2:
                    corpus.append(entry)

    print(f"Loaded {len(corpus)} corpus entries with sentences")

    # ── Subsample with source balance ──
    if args.max_corpus and len(corpus) > args.max_corpus:
        by_source = defaultdict(list)
        for entry in corpus:
            by_source[entry.get("source", "unknown")].append(entry)

        # Proportional sampling per source
        target = args.max_corpus
        ratio = target / len(corpus)
        sampled = []
        for source, entries in by_source.items():
            n = max(1, int(len(entries) * ratio))
            random.shuffle(entries)
            sampled.extend(entries[:n])

        # Trim to exact target
        random.shuffle(sampled)
        corpus = sampled[:target]
        source_counts = Counter(e.get("source", "unknown") for e in corpus)
        print(f"Subsampled to {len(corpus)} entries (source-balanced):")
        for src, cnt in source_counts.most_common():
            print(f"  {src}: {cnt}")


    # ── Phase 1: Text-derived facts (CPU) ──
    print("\n=== Phase 1: Extracting text-derived facts ===")
    all_facts = {}
    for entry in corpus:
        all_facts[entry["id"]] = extract_text_facts(entry)
    print(f"Extracted facts for {len(all_facts)} entries")

    # ── Phase 2: Answer trajectories (vLLM) ──
    trajectories = {}

    if args.trajectory_cache:
        print(f"\n=== Phase 2: Loading cached trajectories from {args.trajectory_cache} ===")
        with open(args.trajectory_cache) as f:
            trajectories = json.load(f)
        print(f"Loaded {len(trajectories)} cached trajectories")

    elif not args.no_vllm:
        print("\n=== Phase 2: Extracting answer trajectories via vLLM ===")
        trajectories = run_trajectory_extraction(
            corpus,
            args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        # Save trajectory cache
        cache_path = Path(args.output).with_suffix(".trajectories.json")
        with open(cache_path, "w") as f:
            json.dump(trajectories, f, indent=2)
        print(f"Saved trajectory cache to {cache_path}")
    else:
        print("\n=== Phase 2: Skipped (--no-vllm) ===")

    # ── Load importance labels (optional) ──
    importance_by_id = {}
    if args.importance:
        print(f"\nLoading importance labels from {args.importance}")
        with open(args.importance) as f:
            for line in f:
                if line.strip():
                    label = json.loads(line)
                    importance_by_id[label["id"]] = label
        print(f"Loaded importance data for {len(importance_by_id)} problems")

    # ── Phase 3: Generate QA pairs ──
    print("\n=== Phase 3: Generating QA pairs ===")
    all_qa = []

    for entry in corpus:
        cid = entry["id"]
        facts = all_facts[cid]
        traj = trajectories.get(cid)
        imp = importance_by_id.get(cid)

        pairs = generate_qa_pairs(entry, facts, trajectory=traj, importance=imp)

        # Limit per entry
        if len(pairs) > args.max_per_entry:
            random.shuffle(pairs)
            pairs = pairs[: args.max_per_entry]

        all_qa.extend(pairs)

    # ── Phase 3b: Word presence QA (programmatic, no LLM) ──
    print("\n=== Phase 3b: Generating word presence QA (programmatic) ===")
    word_qa = []
    for entry in corpus:
        word_qa.extend(generate_word_presence_qa(entry, n_positive=1, n_negative=1))
    print(f"Generated {len(word_qa)} word presence QA pairs (free, no LLM)")
    all_qa.extend(word_qa)

    # ── Phase 4: Conversational QA via LLM (OpenRouter) ──
    if args.openrouter:
        api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY env var or pass --openrouter-key")

        print(f"\n=== Phase 4: Generating conversational QA via OpenRouter ===")
        conv_qa = asyncio.run(
            generate_conversational_qa(
                corpus,
                api_key,
                n_per_entry=args.conv_per_entry,
                concurrency=args.concurrency,
                model=args.openrouter_model,
            )
        )
        all_qa.extend(conv_qa)
    else:
        print("\n=== Phase 4: Skipped (no --openrouter) ===")

    random.shuffle(all_qa)

    # ── Stats ──
    by_type = defaultdict(int)
    by_length = defaultdict(int)
    for qa in all_qa:
        by_type[qa["task_type"]] += 1
        by_length[qa["answer_length"]] += 1

    print(f"\nTotal QA pairs: {len(all_qa)}")
    print(f"\nBy answer length:")
    for length, count in sorted(by_length.items()):
        print(f"  {length}: {count} ({count / len(all_qa) * 100:.1f}%)")

    print(f"\nBy task type:")
    for task, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {task}: {count}")

    # ── Phase 4: Save ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for qa in all_qa:
            f.write(json.dumps(qa) + "\n")

    print(f"\nSaved {len(all_qa)} QA pairs to {output_path}")

    # Trajectory stats
    if trajectories:
        n_with_traj = sum(1 for qa in all_qa if qa["task_type"].startswith(("direct_answer", "answer_at", "first_correct", "answer_changes", "n_answer_changes", "cot_effect", "early_knowledge", "convergence", "trajectory_description")))
        n_text = len(all_qa) - n_with_traj
        print(f"\n  Text-derived QA: {n_text}")
        print(f"  Trajectory QA: {n_with_traj}")


if __name__ == "__main__":
    main()

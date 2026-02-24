"""
Generate on-policy CoT rollouts from Qwen3-8B.

Supports two modes:
1. Local GPU mode: batched generation with local model (original)
2. OpenRouter API mode: async batch calls to qwen/qwen3-8b via OpenRouter

Also supports persona generation (same problems, different system prompts).

Usage:
    # Local GPU mode
    python src/data_pipeline/generate_cots.py --n-problems 500 --output data/cot_corpus/corpus.jsonl

    # OpenRouter API mode (no GPU needed)
    python src/data_pipeline/generate_cots.py --openrouter --n-problems 500 --output data/cot_corpus/corpus.jsonl

    # Generate persona rollouts via OpenRouter
    python src/data_pipeline/generate_cots.py --openrouter --personas --n-problems 1000 \
        --output data/cot_corpus/corpus_persona.jsonl
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import aiohttp
from datasets import load_dataset
from tqdm import tqdm

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from cot_utils import (
    split_cot_into_sentences,
    find_sentence_boundary_positions,
)


# ============================================================
# Problem Sources
# ============================================================

def load_math_problems(n: int, levels: list[str] | None = None) -> list[dict]:
    """Load problems from MATH dataset (train + test splits)."""
    problems = []
    for ds_name in ["hendrycks/competition_mathematics", "HuggingFaceH4/MATH-500"]:
        for split in ["train", "test"]:
            if len(problems) >= n:
                break
            try:
                ds = load_dataset(ds_name, split=split)
            except Exception:
                continue
            for item in ds:
                if levels and item.get("level", "") not in levels:
                    continue
                if "answer" in item and item["answer"]:
                    answer = item["answer"].strip()
                else:
                    solution = item.get("solution", "")
                    boxed = re.findall(r'\\boxed\{([^}]+)\}', solution)
                    if not boxed:
                        continue
                    answer = boxed[-1].strip()
                problems.append({
                    "source": "MATH",
                    "question": item["problem"],
                    "correct_answer": answer,
                    "subject": item.get("subject", item.get("type", "")),
                    "level": item.get("level", ""),
                })
                if len(problems) >= n:
                    break
        if len(problems) >= n:
            break
    return problems


def load_gsm8k_problems(n: int) -> list[dict]:
    """Load problems from GSM8K (train + test splits)."""
    problems = []
    for split in ["train", "test"]:
        if len(problems) >= n:
            break
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for item in ds:
            answer = item["answer"].split("####")[-1].strip()
            problems.append({
                "source": "GSM8K",
                "question": item["question"],
                "correct_answer": answer,
                "subject": "arithmetic",
                "level": "",
            })
            if len(problems) >= n:
                break
    return problems


def load_gpqa_problems(n: int) -> list[dict]:
    """Load problems from GPQA (graduate-level science)."""
    ds = load_dataset("Wanfq/gpqa", "gpqa_diamond", split="train")
    problems = []
    for item in ds:
        question = item.get("Question", "")
        correct_idx = item.get("Correct Answer", "")
        if not question or not correct_idx:
            continue
        problems.append({
            "source": "GPQA",
            "question": question,
            "correct_answer": correct_idx,
            "subject": item.get("Subdomain", "science"),
            "level": "graduate",
        })
        if len(problems) >= n:
            break
    return problems


def load_bbh_problems(n: int) -> list[dict]:
    """Load problems from BIG-Bench Hard (diverse reasoning)."""
    subtasks = [
        "logical_deduction_five_objects", "tracking_shuffled_objects_three_objects",
        "causal_judgement", "date_understanding", "disambiguation_qa",
        "hyperbaton", "navigate", "penguins_in_a_table",
        "reasoning_about_colored_objects", "snarks",
    ]
    problems = []
    for subtask in subtasks:
        if len(problems) >= n:
            break
        try:
            ds = load_dataset("lukaemon/bbh", subtask, split="test")
            for item in ds:
                if len(problems) >= n:
                    break
                question = item.get("input", "")
                answer = item.get("target", "")
                if not question or not answer:
                    continue
                problems.append({
                    "source": "BBH",
                    "question": question,
                    "correct_answer": answer.strip(),
                    "subject": subtask,
                    "level": "",
                })
        except Exception:
            continue
    return problems[:n]


def load_arc_challenge_problems(n: int) -> list[dict]:
    """Load problems from ARC-Challenge (science reasoning)."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    problems = []
    for item in ds:
        question = item.get("question", "")
        choices = item.get("choices", {})
        answer_key = item.get("answerKey", "")
        if not question or not answer_key:
            continue
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if not labels or not texts:
            continue
        choices_text = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
        full_q = f"{question}\n\n{choices_text}\n\nAnswer with the letter."
        problems.append({
            "source": "ARC",
            "question": full_q,
            "correct_answer": answer_key,
            "subject": "science",
            "level": "challenge",
        })
        if len(problems) >= n:
            break
    return problems


def load_strategyqa_problems(n: int) -> list[dict]:
    """Load problems from StrategyQA (multi-hop yes/no reasoning)."""
    ds = load_dataset("tasksource/strategy-qa", split="train")
    problems = []
    for item in ds:
        question = item.get("question", "")
        answer = item.get("answer", None)
        if not question or answer is None:
            continue
        problems.append({
            "source": "StrategyQA",
            "question": question,
            "correct_answer": "yes" if answer else "no",
            "subject": "multi-hop",
            "level": "",
        })
        if len(problems) >= n:
            break
    return problems


def load_drop_problems(n: int) -> list[dict]:
    """Load problems from DROP (discrete reasoning over paragraphs)."""
    ds = load_dataset("ucinlp/drop", split="validation")
    problems = []
    for item in ds:
        passage = item.get("passage", "")
        question = item.get("question", "")
        answers = item.get("answers_spans", {})
        if not passage or not question:
            continue
        spans = answers.get("spans", [])
        if not spans:
            continue
        answer = spans[0]
        full_q = f"Passage: {passage[:500]}\n\nQuestion: {question}"
        problems.append({
            "source": "DROP",
            "question": full_q,
            "correct_answer": answer,
            "subject": "reading_comprehension",
            "level": "",
        })
        if len(problems) >= n:
            break
    return problems


def load_logiqa_problems(n: int) -> list[dict]:
    """Load problems from LogiQA (formal logic puzzles)."""
    ds = load_dataset("lucasmccabe/logiqa", split="test", revision="refs/convert/parquet")
    problems = []
    for item in ds:
        context = item.get("context", "")
        question = item.get("query", "")
        options = item.get("options", [])
        answer_idx = item.get("correct_option", 0)
        if not context or not question or not options:
            continue
        labels = ["A", "B", "C", "D"]
        choices_text = "\n".join(
            f"{labels[i]}) {opt}" for i, opt in enumerate(options) if i < 4
        )
        full_q = f"{context}\n\n{question}\n\n{choices_text}\n\nAnswer with the letter."
        correct_letter = labels[answer_idx] if answer_idx < len(labels) else "A"
        problems.append({
            "source": "LogiQA",
            "question": full_q,
            "correct_answer": correct_letter,
            "subject": "logic",
            "level": "",
        })
        if len(problems) >= n:
            break
    return problems


def load_mmlu_pro_problems(n: int) -> list[dict]:
    """Load problems from MMLU-Pro (multi-domain MCQ)."""
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    problems = []
    for item in ds:
        question = item.get("question", "")
        options = item.get("options", [])
        answer = item.get("answer", "")
        category = item.get("category", "")
        if not question or not options or not answer:
            continue
        labels = [chr(ord('A') + i) for i in range(len(options))]
        choices_text = "\n".join(f"{l}) {o}" for l, o in zip(labels, options))
        full_q = f"{question}\n\n{choices_text}\n\nAnswer with the letter."
        problems.append({
            "source": "MMLU-Pro",
            "question": full_q,
            "correct_answer": answer,
            "subject": category,
            "level": "",
        })
        if len(problems) >= n:
            break
    return problems


def load_commonsenseqa_problems(n: int) -> list[dict]:
    """Load problems from CommonsenseQA (train + validation)."""
    problems = []
    for split in ["train", "validation"]:
        if len(problems) >= n:
            break
        ds = load_dataset("tau/commonsense_qa", split=split)
        for item in ds:
            question = item.get("question", "")
            choices = item.get("choices", {})
            answer_key = item.get("answerKey", "")
            if not question or not answer_key:
                continue
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            if not labels or not texts:
                continue
            choices_text = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
            full_q = f"{question}\n\n{choices_text}\n\nAnswer with the letter."
            problems.append({
                "source": "CommonsenseQA",
                "question": full_q,
                "correct_answer": answer_key,
                "subject": "commonsense",
                "level": "",
            })
            if len(problems) >= n:
                break
    return problems


def load_aqua_rat_problems(n: int) -> list[dict]:
    """Load problems from AQUA-RAT (algebraic word problems, train + test)."""
    problems = []
    for split in ["train", "test"]:
        if len(problems) >= n:
            break
        ds = load_dataset("deepmind/aqua_rat", "raw", split=split)
        for item in ds:
            question = item.get("question", "")
            options = item.get("options", [])
            correct = item.get("correct", "")
            if not question or not options or not correct:
                continue
            choices_text = "\n".join(options)
            full_q = f"{question}\n\n{choices_text}\n\nAnswer with the letter."
            problems.append({
                "source": "AQUA-RAT",
                "question": full_q,
                "correct_answer": correct,
                "subject": "algebra",
                "level": "",
            })
            if len(problems) >= n:
                break
    return problems


def load_medqa_problems(n: int) -> list[dict]:
    """Load problems from MedQA (medical reasoning)."""
    # Try multiple MedQA dataset sources
    for ds_name, config, split in [
        ("GBaker/MedQA-USMLE-4-options", None, "test"),
        ("bigbio/med_qa", "med_qa_en_source", "test"),
    ]:
        try:
            if config:
                ds = load_dataset(ds_name, config, split=split)
            else:
                ds = load_dataset(ds_name, split=split)
            break
        except Exception:
            continue
    else:
        print("  MedQA: Could not load from any known source, returning empty")
        return []

    problems = []
    for item in ds:
        # Handle different MedQA formats
        question = item.get("question", "")
        if not question:
            continue

        # bigbio format or GBaker dict format
        if "options" in item:
            options = item["options"]
            if isinstance(options, dict):
                # GBaker/MedQA-USMLE-4-options: options is {"A": "...", "B": "...", ...}
                labels = sorted(options.keys())
                texts = [options[l] for l in labels]
                choices_text = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
                full_q = f"{question}\n\n{choices_text}\n\nAnswer with the letter."
                answer = item.get("answer_idx", item.get("answer", ""))
            elif isinstance(options, list) and len(options) > 0:
                if isinstance(options[0], dict):
                    labels = [o.get("key", chr(ord('A') + i)) for i, o in enumerate(options)]
                    texts = [o.get("value", "") for o in options]
                else:
                    labels = [chr(ord('A') + i) for i in range(len(options))]
                    texts = options
                choices_text = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
                full_q = f"{question}\n\n{choices_text}\n\nAnswer with the letter."
                answer = item.get("answer_idx", item.get("answer", ""))
                if isinstance(answer, int):
                    answer = labels[answer] if answer < len(labels) else ""
            else:
                continue
        # GBaker legacy format with op1/op2/op3/op4
        elif "op1" in item:
            options = [item.get(f"op{i}", "") for i in range(1, 5)]
            labels = ["A", "B", "C", "D"]
            choices_text = "\n".join(f"{l}) {t}" for l, t in zip(labels, options))
            full_q = f"{question}\n\n{choices_text}\n\nAnswer with the letter."
            answer = item.get("answer_idx", item.get("answer", ""))
            if isinstance(answer, int):
                answer = labels[answer] if answer < len(labels) else ""
        else:
            continue

        if not answer:
            continue

        problems.append({
            "source": "MedQA",
            "question": full_q,
            "correct_answer": str(answer),
            "subject": "medical",
            "level": "",
        })
        if len(problems) >= n:
            break
    return problems


def load_asdiv_problems(n: int) -> list[dict]:
    """Load problems from ASDiv (diverse arithmetic)."""
    ds = load_dataset("EleutherAI/asdiv", split="validation")
    problems = []
    for item in ds:
        body = item.get("body", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        if not body or not question or not answer:
            continue
        # Strip units in parentheses: "9 (apples)" -> "9"
        answer_clean = re.sub(r'\s*\(.*?\)\s*$', '', answer).strip()
        full_q = f"{body} {question}"
        problems.append({
            "source": "ASDiv",
            "question": full_q,
            "correct_answer": answer_clean,
            "subject": item.get("solution_type", "arithmetic"),
            "level": "",
        })
        if len(problems) >= n:
            break
    return problems


def load_scienceqa_problems(n: int) -> list[dict]:
    """Load text-only problems from ScienceQA."""
    problems = []
    for split in ["test", "train"]:
        if len(problems) >= n:
            break
        ds = load_dataset("derek-thomas/ScienceQA", split=split)
        for item in ds:
            # Skip image-based questions
            if item.get("image") is not None:
                continue
            question = item.get("question", "")
            choices = item.get("choices", [])
            answer_idx = item.get("answer", None)
            if not question or not choices or answer_idx is None:
                continue
            labels = [chr(ord('A') + i) for i in range(len(choices))]
            choices_text = "\n".join(f"{l}) {c}" for l, c in zip(labels, choices))
            full_q = f"{question}\n\n{choices_text}\n\nAnswer with the letter."
            correct_letter = labels[answer_idx] if answer_idx < len(labels) else "A"
            problems.append({
                "source": "ScienceQA",
                "question": full_q,
                "correct_answer": correct_letter,
                "subject": item.get("subject", "science"),
                "level": item.get("grade", ""),
            })
            if len(problems) >= n:
                break
    return problems


def load_arc_easy_problems(n: int) -> list[dict]:
    """Load problems from ARC-Easy."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    problems = []
    for item in ds:
        question = item.get("question", "")
        choices = item.get("choices", {})
        answer_key = item.get("answerKey", "")
        if not question or not answer_key:
            continue
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if not labels or not texts:
            continue
        choices_text = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
        full_q = f"{question}\n\n{choices_text}\n\nAnswer with the letter."
        problems.append({
            "source": "ARC-Easy",
            "question": full_q,
            "correct_answer": answer_key,
            "subject": "science",
            "level": "easy",
        })
        if len(problems) >= n:
            break
    return problems


def load_scruples_problems(n: int) -> list[dict]:
    """Load ethical judgment problems from Scruples (AITA-style).

    Ground truth is binarized human judgment: RIGHT or WRONG.
    """
    ds = load_dataset("metaeval/scruples", split="train", streaming=True)
    problems = []
    for item in ds:
        title = item.get("title", "")
        text = item.get("text", "")
        label = item.get("binarized_label", "")
        if not text or not label:
            continue
        # Skip very long posts (>300 words) to keep input reasonable
        if len(text.split()) > 300:
            continue
        full_q = f"{title}\n\n{text}\n\nBased on this situation, is the author in the right or in the wrong? Answer with RIGHT or WRONG."
        problems.append({
            "source": "Scruples",
            "question": full_q,
            "correct_answer": label,
            "subject": "ethics",
            "level": "",
        })
        if len(problems) >= n:
            break
    print(f"    Scruples: loaded {len(problems)}")
    return problems


def load_lmsys_problems(n: int) -> list[dict]:
    """Load filtered questions from LMSYS-Chat-1M.

    Filters: English, 15-60 words, no jailbreak/roleplay/generation/code.
    No ground truth answer — fields will be null.

    Uses streaming with a progress counter so we know it's working.
    For large n, scans proportionally more rows (~3.5x due to ~29% filter pass rate).
    """
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)

    generation_keywords = ['write', 'generate', 'create', 'give me', 'compose', 'draft',
                          'tell me a story', 'poem', 'essay', 'introduction for']
    jailbreak_keywords = ['you are the text completion', 'you must complete', 'ignore previous',
                         'pretend you', 'act as', 'roleplay', 'dan ']

    problems = []
    scanned = 0
    for row in ds:
        if len(problems) >= n:
            break
        scanned += 1
        if scanned % 1000 == 0:
            print(f"    LMSYS: scanned {scanned}, kept {len(problems)}/{n}")

        lang = row.get("language", "unknown")
        if lang != "English":
            continue

        conv = row["conversation"]
        first_user = None
        for msg in conv:
            if msg["role"] == "user":
                first_user = msg["content"]
                break
        if not first_user:
            continue

        words = first_user.split()
        if len(words) < 15 or len(words) > 60:
            continue

        lower = first_user.lower()

        if any(kw in lower for kw in jailbreak_keywords):
            continue
        if '```' in first_user or 'def ' in first_user or 'function(' in first_user:
            continue
        if any(lower.startswith(kw) or f' {kw} ' in lower[:50] for kw in generation_keywords):
            continue

        problems.append({
            "source": "LMSYS",
            "question": first_user,
            "correct_answer": None,
            "subject": "diverse",
            "level": "",
        })

    print(f"    LMSYS: scanned {scanned} total, kept {len(problems)}")
    return problems


DATASET_LOADERS = {
    "math": lambda n, levels: load_math_problems(n, levels),
    "gsm8k": lambda n, levels: load_gsm8k_problems(n),
    "gpqa": lambda n, levels: load_gpqa_problems(n),
    "bbh": lambda n, levels: load_bbh_problems(n),
    "arc_challenge": lambda n, levels: load_arc_challenge_problems(n),
    "arc_easy": lambda n, levels: load_arc_easy_problems(n),
    "strategyqa": lambda n, levels: load_strategyqa_problems(n),
    "drop": lambda n, levels: load_drop_problems(n),
    "logiqa": lambda n, levels: load_logiqa_problems(n),
    "mmlu_pro": lambda n, levels: load_mmlu_pro_problems(n),
    "commonsenseqa": lambda n, levels: load_commonsenseqa_problems(n),
    "aqua_rat": lambda n, levels: load_aqua_rat_problems(n),
    "medqa": lambda n, levels: load_medqa_problems(n),
    "asdiv": lambda n, levels: load_asdiv_problems(n),
    "scienceqa": lambda n, levels: load_scienceqa_problems(n),
    "scruples": lambda n, levels: load_scruples_problems(n),
    "lmsys": lambda n, levels: load_lmsys_problems(n),
}

# Domain mapping: source name -> domain label (for Task 4: domain classification)
SOURCE_TO_DOMAIN = {
    "MATH": "math",
    "GSM8K": "math",
    "GPQA": "science",
    "BBH": "logic",
    "ARC": "science",
    "ARC-Easy": "science",
    "StrategyQA": "commonsense",
    "DROP": "reading",
    "LogiQA": "logic",
    "MMLU-Pro": "multi_domain",
    "CommonsenseQA": "commonsense",
    "AQUA-RAT": "math",
    "MedQA": "medical",
    "ASDiv": "math",
    "ScienceQA": "science",
    "Scruples": "ethics",
    "LMSYS": "diverse",
}


def load_all_problems(
    sources: list[str],
    n_per_source: int | dict[str, int],
    levels: list[str] | None = None,
) -> list[dict]:
    """Load problems from specified sources.

    n_per_source can be an int (same for all) or a dict mapping source name to count.
    """
    problems = []
    for source in sources:
        loader = DATASET_LOADERS.get(source)
        if loader is None:
            raise ValueError(f"Unknown source: {source}. Available: {list(DATASET_LOADERS.keys())}")
        n = n_per_source[source] if isinstance(n_per_source, dict) else n_per_source
        try:
            loaded = loader(n, levels)
            problems.extend(loaded)
            print(f"  {source}: loaded {len(loaded)} problems")
        except Exception as e:
            print(f"  {source}: FAILED to load ({e}), skipping")
    print(f"Loaded {len(problems)} total problems from {sources}")
    return problems


# ============================================================
# Answer Extraction
# ============================================================

def extract_answer(response: str) -> str | None:
    """Extract final answer from model response."""
    if not response:
        return None
    boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
    if boxed:
        return boxed[-1].strip()
    hashes = re.findall(r'####\s*(.+?)(?:\n|$)', response)
    if hashes:
        return hashes[-1].strip().replace(",", "")
    # Letter answers (for MCQ)
    letter_match = re.findall(r'(?:answer|option)\s+(?:is\s+)?([A-E])\b', response, re.IGNORECASE)
    if letter_match:
        return letter_match[-1].upper()
    # "The answer is X" pattern
    answer_is = re.findall(r'(?:the\s+)?answer\s+is\s*:?\s*(.+?)(?:\.|$)', response, re.IGNORECASE)
    if answer_is:
        ans = answer_is[-1].strip()
        if len(ans) <= 50:
            return ans
    answer_patterns = [
        r'(?:answer|result)\s+(?:is|=)\s*[:\s]*(-?\d+(?:[.,]\d+)*)',
        r'(?:=|equals?)\s*(-?\d+(?:[.,]\d+)*)\s*$',
        r'\*\*(-?\d+(?:[.,]\d+)*)\*\*\s*$',
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].strip().replace(",", "")
    numbers = re.findall(r'(-?\d+(?:\.\d+)?)', response)
    if numbers:
        return numbers[-1]
    return None


def answers_match(model_answer: str | None, correct_answer: str) -> bool:
    """Check if model answer matches correct answer (fuzzy numeric comparison)."""
    if model_answer is None:
        return False
    def normalize(s: str) -> str:
        s = s.strip().replace(",", "").replace(" ", "")
        if s.endswith(".0"):
            s = s[:-2]
        frac = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', s)
        if frac:
            s = str(int(frac.group(1)) / int(frac.group(2)))
        return s.lower()
    return normalize(model_answer) == normalize(correct_answer)


# ============================================================
# Batched Generation (Local GPU)
# ============================================================

def batch_generate(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    """Generate responses for a batch of prompts."""
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    prompt_lens = [inputs["attention_mask"][i].sum().item() for i in range(len(prompts))]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    responses = []
    for i in range(len(prompts)):
        gen_ids = outputs[i][prompt_lens[i]:]
        responses.append(tokenizer.decode(gen_ids, skip_special_tokens=False))
    return responses


# ============================================================
# OpenRouter API Mode
# ============================================================

PERSONAS = {
    "math_tutor": "You are a patient math tutor who explains step by step.",
    "physics_professor": "You are a physics professor at a top university. Be precise and rigorous.",
    "student": "You are a student working through this problem for homework. Think aloud.",
    "formal_expert": "You are a formal academic expert. Use precise notation and terminology.",
    "casual_explainer": "You are explaining this to a friend in a casual, conversational way.",
    "skeptical_scientist": "You are a skeptical scientist who double-checks every step.",
    "step_by_step_solver": "You are a solver who breaks every problem into tiny explicit steps.",
    "default": None,  # No system prompt
}


_completed_count = 0
_failed_count = 0
_total_count = 0


async def openrouter_generate_single(
    session,
    question: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    enable_thinking: bool = True,
) -> dict:
    """Single OpenRouter API call. Returns dict with 'reasoning' and 'content' fields."""
    global _completed_count, _failed_count
    async with semaphore:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        body = {
            "model": "qwen/qwen3-8b",
            "messages": messages,
            "temperature": 0.6,
            "top_p": 0.95,
            # OpenRouter reasoning controls:
            # enable_thinking=True keeps reasoning content, False requests direct answers.
            "reasoning": {"enabled": bool(enable_thinking), "exclude": not enable_thinking},
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

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
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    msg = data["choices"][0]["message"]
                    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                    content = msg.get("content", "")
                    _completed_count += 1
                    if _completed_count % 25 == 0:
                        print(f"  Progress: {_completed_count}/{_total_count} done, {_failed_count} failed")
                    return {"reasoning": reasoning, "content": content}
            except asyncio.TimeoutError:
                if attempt == 2:
                    _failed_count += 1
                    return {"reasoning": "", "content": ""}
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 2:
                    _failed_count += 1
                    if _failed_count <= 10:
                        print(f"  OpenRouter error: {e}")
                    return {"reasoning": "", "content": ""}
                await asyncio.sleep(2 ** attempt)
        return {"reasoning": "", "content": ""}


async def openrouter_generate_batch(
    problems: list[dict],
    api_key: str,
    concurrency: int = 50,
    system_prompt: str | None = None,
    persona_name: str | None = None,
    n_rollouts: int = 1,
    skip_direct_sources: set[str] | None = None,
) -> tuple[list[list[dict]], list[dict]]:
    """Generate CoT and direct responses for all problems via OpenRouter.

    Returns (cot_responses_per_rollout, direct_responses).
    Each response is a dict with 'reasoning' (CoT text) and 'content' (answer text).
    cot_responses_per_rollout: [rollout_idx][problem_idx] -> dict.
    direct_responses: generated once per problem (skipped for sources in skip_direct_sources).
    """
    global _completed_count, _failed_count, _total_count
    semaphore = asyncio.Semaphore(concurrency)
    if skip_direct_sources is None:
        skip_direct_sources = set()

    empty_response = {"reasoning": "", "content": ""}
    all_cot_rollouts: list[list[dict]] = []

    async with aiohttp.ClientSession() as session:
        for rollout_idx in range(n_rollouts):
            _completed_count = 0
            _failed_count = 0
            _total_count = len(problems)
            print(f"  Generating CoT rollout {rollout_idx + 1}/{n_rollouts} ({len(problems)} problems)...")
            cot_responses = list(await asyncio.gather(*[
                openrouter_generate_single(
                    session, p["question"], api_key, semaphore,
                    system_prompt=system_prompt,
                    max_tokens=16384, enable_thinking=True,
                )
                for p in problems
            ]))
            print(f"  Rollout {rollout_idx + 1} done: {_completed_count} succeeded, {_failed_count} failed")
            all_cot_rollouts.append(cot_responses)

        # Direct (no-thinking) generation — once per problem, skip for LMSYS etc.
        _completed_count = 0
        _failed_count = 0
        problems_needing_direct = [
            p for p in problems if p["source"] not in skip_direct_sources
        ]
        direct_responses_map: dict[int, dict] = {}
        if problems_needing_direct:
            print(f"  Generating {len(problems_needing_direct)} direct responses (skipping {len(problems) - len(problems_needing_direct)} without ground truth)...")
            direct_results = list(await asyncio.gather(*[
                openrouter_generate_single(
                    session, p["question"], api_key, semaphore,
                    system_prompt=system_prompt,
                    max_tokens=200, enable_thinking=False,
                )
                for p in problems_needing_direct
            ]))
            print(f"  Direct done: {_completed_count} succeeded, {_failed_count} failed")
            # Map back to original indices
            direct_idx = 0
            for i, p in enumerate(problems):
                if p["source"] not in skip_direct_sources:
                    direct_responses_map[i] = direct_results[direct_idx]
                    direct_idx += 1

        direct_responses = [direct_responses_map.get(i, empty_response) for i in range(len(problems))]

    return all_cot_rollouts, direct_responses


# ============================================================
# Process Results
# ============================================================

def process_results_openrouter(
    problems: list[dict],
    all_cot_rollouts: list[list[dict]],
    direct_responses: list[dict],
    tokenizer=None,
    persona: str | None = None,
) -> list[dict]:
    """Post-process OpenRouter responses into corpus entries.

    all_cot_rollouts: list of rollouts, each a list of response dicts per problem.
    Each response dict has 'reasoning' (CoT text) and 'content' (answer text).
    If tokenizer is provided, computes boundary_positions.
    """
    results = []
    n_rollouts = len(all_cot_rollouts)

    for i, problem in enumerate(problems):
        question = problem["question"]
        has_ground_truth = problem["correct_answer"] is not None

        # Direct response: extract content (no reasoning for direct calls)
        direct_result = direct_responses[i]
        direct_content = direct_result["content"] if isinstance(direct_result, dict) else str(direct_result)

        if has_ground_truth:
            direct_answer = extract_answer(direct_content)
            direct_correct = answers_match(direct_answer, problem["correct_answer"])
        else:
            direct_answer = None
            direct_correct = None

        for rollout_idx in range(n_rollouts):
            cot_result = all_cot_rollouts[rollout_idx][i]
            # Extract reasoning (the actual CoT) and content (the answer after thinking)
            cot_reasoning = cot_result["reasoning"] if isinstance(cot_result, dict) else str(cot_result)
            cot_content = cot_result["content"] if isinstance(cot_result, dict) else ""

            if not cot_reasoning and not cot_content:
                continue

            if has_ground_truth:
                # Extract answer from content (the post-thinking answer), not from reasoning
                cot_answer = extract_answer(cot_content) if cot_content else extract_answer(cot_reasoning)
                cot_correct = answers_match(cot_answer, problem["correct_answer"])
                if cot_correct and not direct_correct:
                    category = "load_bearing"
                elif cot_correct and direct_correct:
                    category = "both_correct"
                elif not cot_correct and not direct_correct:
                    category = "both_wrong"
                else:
                    category = "cot_hurt"
            else:
                cot_answer = None
                cot_correct = None
                category = None

            # cot_response = the actual chain-of-thought reasoning
            cot_response = cot_reasoning

            sentences = split_cot_into_sentences(cot_response)
            if len(sentences) < 2:
                continue

            # boundary_positions deferred — too slow during generation.
            # Compute later with: find_sentence_boundary_positions(tokenizer, full_text, sentences)
            boundary_positions = []

            domain = SOURCE_TO_DOMAIN.get(problem["source"], "unknown")

            entry = {
                "id": f"{problem['source'].lower()}_{i:04d}_r{rollout_idx}",
                "source": problem["source"],
                "domain": domain,
                "question": question,
                "correct_answer": problem["correct_answer"],
                "subject": problem.get("subject", ""),
                "level": problem.get("level", ""),
                "cot_response": cot_response,
                "cot_content": cot_content,
                "direct_response": direct_content if rollout_idx == 0 else None,
                "cot_answer": cot_answer,
                "direct_answer": direct_answer if rollout_idx == 0 else None,
                "cot_correct": cot_correct,
                "direct_correct": direct_correct if rollout_idx == 0 else None,
                "category": category,
                "sentences": sentences,
                "boundary_positions": boundary_positions,
                "n_sentences": len(sentences),
                "rollout_idx": rollout_idx,
            }

            if persona:
                entry["persona"] = persona

            results.append(entry)

    return results


def process_results(
    problems: list[dict],
    cot_responses: list[str],
    direct_responses: list[str],
    tokenizer,
) -> list[dict]:
    """Post-process generated responses into corpus entries (local GPU mode)."""
    results = []
    for i, (problem, cot_response, direct_response) in enumerate(
        zip(problems, cot_responses, direct_responses)
    ):
        question = problem["question"]
        cot_answer = extract_answer(cot_response)
        direct_answer = extract_answer(direct_response)
        cot_correct = answers_match(cot_answer, problem["correct_answer"])
        direct_correct = answers_match(direct_answer, problem["correct_answer"])

        if cot_correct and not direct_correct:
            category = "load_bearing"
        elif cot_correct and direct_correct:
            category = "both_correct"
        elif not cot_correct and not direct_correct:
            category = "both_wrong"
        else:
            category = "cot_hurt"

        sentences = split_cot_into_sentences(cot_response)
        if len(sentences) < 2:
            continue

        # boundary_positions deferred — too slow during generation.
        boundary_positions = []

        domain = SOURCE_TO_DOMAIN.get(problem["source"], "unknown")

        results.append({
            "id": f"{problem['source'].lower()}_{i:04d}",
            "source": problem["source"],
            "domain": domain,
            "question": question,
            "correct_answer": problem["correct_answer"],
            "subject": problem.get("subject", ""),
            "level": problem.get("level", ""),
            "cot_response": cot_response,
            "direct_response": direct_response,
            "cot_answer": cot_answer,
            "direct_answer": direct_answer,
            "cot_correct": cot_correct,
            "direct_correct": direct_correct,
            "category": category,
            "sentences": sentences,
            "boundary_positions": boundary_positions,
            "n_sentences": len(sentences),
        })

    return results


# ============================================================
# Preset corpus configurations
# ============================================================

# Full scale: 125K problems × 2 rollouts ≈ 200M CoT tokens (~$120)
FULL_SPLIT = {
    "math": 8000,
    "gsm8k": 8800,
    "aqua_rat": 25000,
    "asdiv": 2300,
    "arc_challenge": 1172,
    "arc_easy": 2376,
    "commonsenseqa": 10000,
    "scienceqa": 4200,
    "medqa": 1273,
    "mmlu_pro": 12000,
    "scruples": 12000,
    "lmsys": 37927,
}

# Medium scale: 50K problems × 1 rollout ≈ 97M CoT tokens (~$25)
MEDIUM_SPLIT = {
    "math": 3200,
    "gsm8k": 3520,
    "aqua_rat": 10000,
    "asdiv": 920,
    "arc_challenge": 469,
    "arc_easy": 950,
    "commonsenseqa": 4000,
    "scienceqa": 1680,
    "medqa": 509,
    "mmlu_pro": 4800,
    "scruples": 5000,
    "lmsys": 14971,
}

# Mini scale: 1/200th for testing (~625 problems × 2 rollouts ≈ 500K CoT tokens)
MINI_SPLIT = {
    "math": 40,
    "gsm8k": 44,
    "aqua_rat": 125,
    "asdiv": 12,
    "arc_challenge": 9,
    "arc_easy": 9,
    "commonsenseqa": 50,
    "scienceqa": 21,
    "medqa": 7,
    "mmlu_pro": 60,
    "scruples": 25,
    "lmsys": 250,
}

# Sources without ground truth answers (skip direct response generation)
NO_GROUND_TRUTH_SOURCES = {"LMSYS"}


# ============================================================
# Main
# ============================================================

def _is_blackwell_gpu() -> bool:
    """Best-effort Blackwell detection (compute capability >= 12)."""
    try:
        import torch
    except Exception:
        return False

    if not torch.cuda.is_available():
        return False

    try:
        majors = [torch.cuda.get_device_capability(i)[0] for i in range(torch.cuda.device_count())]
    except Exception:
        return False

    return bool(majors) and max(majors) >= 12


def _should_enforce_vllm_eager(args) -> bool:
    """Decide whether to set vLLM enforce_eager=True."""
    import os

    if getattr(args, "vllm_enforce_eager", False):
        return True
    if getattr(args, "no_vllm_eager_auto", False):
        return False

    if os.environ.get("COT_ORACLE_VLLM_ENFORCE_EAGER") == "1":
        return True
    if os.environ.get("COT_ORACLE_VLLM_NO_EAGER_AUTO") == "1":
        return False

    return _is_blackwell_gpu()

def main():
    parser = argparse.ArgumentParser(description="Generate CoT corpus v5")
    parser.add_argument("--preset", choices=["full", "medium", "mini"], default=None,
                        help="Use preset split (full=125K, medium=50K, mini=625 problems)")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Override sources (default: from preset or all)")
    parser.add_argument("--n-problems", type=int, default=500,
                        help="Problems per source (ignored if --preset is used)")
    parser.add_argument("--n-rollouts", type=int, default=2,
                        help="Number of CoT rollouts per problem (default: 2)")
    parser.add_argument("--levels", nargs="*", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", default=None,
                        help="Output path (default: auto from preset)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--engine", choices=["auto", "vllm", "sglang"], default="auto",
                        help="Inference engine for local GPU mode (default: auto = try vLLM then SGLang)")
    parser.add_argument("--vllm-enforce-eager", action="store_true",
                        help="Force vLLM enforce_eager=True (recommended on Blackwell/RTX 5090)")
    parser.add_argument("--no-vllm-eager-auto", action="store_true",
                        help="Disable automatic Blackwell -> enforce_eager behavior for vLLM")
    # OpenRouter mode
    parser.add_argument("--openrouter", action="store_true",
                        help="Use OpenRouter API instead of local GPU")
    parser.add_argument("--openrouter-key", default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--concurrency", type=int, default=50,
                        help="Max concurrent OpenRouter requests")
    # Persona mode
    parser.add_argument("--personas", action="store_true",
                        help="Generate persona rollouts (multiple system prompts per problem)")
    parser.add_argument("--persona-list", nargs="+", default=None,
                        help="Specific personas to use (default: all)")
    args = parser.parse_args()

    # Determine split and output path
    if args.preset == "full":
        split = FULL_SPLIT
        default_output = "data/cot_corpus_v5/corpus.jsonl"
    elif args.preset == "medium":
        split = MEDIUM_SPLIT
        default_output = "data/cot_corpus_v5/corpus_medium.jsonl"
    elif args.preset == "mini":
        split = MINI_SPLIT
        default_output = "data/cot_corpus_v5/mini_corpus.jsonl"
    else:
        split = args.n_problems
        default_output = "data/cot_corpus_v5/corpus.jsonl"

    output_path = Path(args.output or default_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine sources
    if args.sources:
        sources = args.sources
    elif isinstance(split, dict):
        sources = list(split.keys())
    else:
        sources = list(DATASET_LOADERS.keys())

    # Load problems
    print(f"Loading problems ({args.preset or 'custom'} preset, {args.n_rollouts} rollouts)...")
    problems = load_all_problems(sources, split, args.levels)

    if args.openrouter:
        import os
        api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY env var or pass --openrouter-key")

        # Try loading tokenizer for boundary position computation (no GPU needed).
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer for {args.model}...")
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            print("Tokenizer loaded — will compute boundary_positions")
        except Exception as e:
            print(f"Tokenizer not available ({e}) — boundary_positions deferred to GPU")

        if args.personas:
            persona_names = args.persona_list or list(PERSONAS.keys())
            all_results = []

            for persona_name in persona_names:
                system_prompt = PERSONAS.get(persona_name)
                print(f"\n{'=' * 60}")
                print(f"Generating for persona: {persona_name}")
                print(f"{'=' * 60}")

                all_cot_rollouts, direct_responses = asyncio.run(
                    openrouter_generate_batch(
                        problems, api_key,
                        concurrency=args.concurrency,
                        system_prompt=system_prompt,
                        persona_name=persona_name,
                        n_rollouts=args.n_rollouts,
                        skip_direct_sources=NO_GROUND_TRUTH_SOURCES,
                    )
                )

                results = process_results_openrouter(
                    problems, all_cot_rollouts, direct_responses,
                    tokenizer=tokenizer, persona=persona_name,
                )
                all_results.extend(results)
                print(f"  {persona_name}: {len(results)} entries")

            saved = 0
            with open(output_path, "w") as f:
                for r in all_results:
                    f.write(json.dumps(r) + "\n")
                    saved += 1

            print(f"\nTotal persona entries: {saved}")
            print(f"Output: {output_path}")

        else:
            # Standard OpenRouter mode
            all_cot_rollouts, direct_responses = asyncio.run(
                openrouter_generate_batch(
                    problems, api_key,
                    concurrency=args.concurrency,
                    n_rollouts=args.n_rollouts,
                    skip_direct_sources=NO_GROUND_TRUTH_SOURCES,
                )
            )

            results = process_results_openrouter(
                problems, all_cot_rollouts, direct_responses, tokenizer=tokenizer,
            )

            # Print stats
            stats: dict[str, int] = {}
            source_counts: dict[str, int] = {}
            saved = 0
            with open(output_path, "w") as f:
                for r in results:
                    cat = r["category"] or "no_ground_truth"
                    stats[cat] = stats.get(cat, 0) + 1
                    source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1
                    f.write(json.dumps(r) + "\n")
                    saved += 1

            total_cot_chars = sum(len(r.get("cot_response", "")) for r in results)

            print(f"\n{'=' * 60}")
            print(f"FINAL STATS")
            print(f"{'=' * 60}")
            print(f"  Problems loaded: {len(problems)}")
            print(f"  Rollouts per problem: {args.n_rollouts}")
            print(f"  Total entries saved: {saved}")
            print(f"  Est. CoT tokens: {total_cot_chars // 4:,}")
            print(f"\n  By category:")
            for k, v in sorted(stats.items()):
                pct = v / saved * 100 if saved > 0 else 0
                print(f"    {k}: {v} ({pct:.1f}%)")
            print(f"\n  By source:")
            for k, v in sorted(source_counts.items(), key=lambda x: -x[1]):
                print(f"    {k}: {v}")
            print(f"\n  Output: {output_path}")

    else:
        # Local GPU mode — try vLLM first, fall back to SGLang
        from transformers import AutoTokenizer
        import time

        print(f"Loading tokenizer for {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        engine_choice = getattr(args, "engine", "auto")
        use_sglang = False

        if engine_choice in ("auto", "vllm"):
            try:
                from vllm import LLM, SamplingParams
                print(f"Loading {args.model} with vLLM (bf16)...")
                llm_kwargs = dict(
                    model=args.model,
                    dtype="bfloat16",
                    tensor_parallel_size=1,
                    max_model_len=8192,
                    gpu_memory_utilization=0.95,
                    enable_prefix_caching=True,
                )
                eager_mode = _should_enforce_vllm_eager(args)
                if eager_mode:
                    llm_kwargs["enforce_eager"] = True
                    print("vLLM eager mode enabled (Blackwell-safe).")
                try:
                    llm = LLM(**llm_kwargs)
                except TypeError as te:
                    if eager_mode and "enforce_eager" in str(te):
                        print("vLLM build does not support enforce_eager; retrying without it.")
                        llm_kwargs.pop("enforce_eager", None)
                        llm = LLM(**llm_kwargs)
                    else:
                        raise
                # Quick smoke test to catch PTX/CUDA errors early
                from vllm import SamplingParams as _SP
                _test = llm.generate(["test"], _SP(max_tokens=1))
                del _test
                cot_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=4096)
                direct_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=200)
                print("vLLM loaded and verified.")
            except Exception as e:
                if engine_choice == "vllm":
                    raise
                print(f"vLLM failed ({e}), falling back to SGLang...")
                use_sglang = True

        if engine_choice == "sglang" or use_sglang:
            use_sglang = True
            import sglang as sgl
            print(f"Loading {args.model} with SGLang (bf16)...")
            llm = sgl.Engine(
                model_path=args.model,
                dtype="bfloat16",
                mem_fraction_static=0.85,
            )
            cot_params = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 4096}
            direct_params = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 200}
            print("SGLang loaded successfully.")

        # Format all prompts
        print("Formatting prompts...")
        cot_prompts = []
        direct_prompts = []
        direct_indices = []
        for i, p in enumerate(problems):
            msgs = [{"role": "user", "content": p["question"]}]
            cot_prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            ))
            if p["source"] not in NO_GROUND_TRUTH_SOURCES:
                direct_prompts.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                ))
                direct_indices.append(i)

        def run_generate(prompts, params):
            """Run generation with either vLLM or SGLang."""
            if use_sglang:
                outputs = llm.generate(prompts, sampling_params=params)
                return [o["text"] for o in outputs]
            else:
                outputs = llm.generate(prompts, params)
                return [o.outputs[0].text for o in outputs]

        def count_tokens(texts):
            return sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)

        n_rollouts = args.n_rollouts
        all_cot_rollouts: list[list[dict]] = []

        for rollout_idx in range(n_rollouts):
            print(f"\n{'=' * 60}")
            print(f"CoT rollout {rollout_idx + 1}/{n_rollouts}: {len(cot_prompts)} prompts")
            print(f"{'=' * 60}")
            t0 = time.time()
            raw_texts = run_generate(cot_prompts, cot_params)
            elapsed = time.time() - t0
            total_toks = count_tokens(raw_texts)
            print(f"  Done in {elapsed:.1f}s — {total_toks:,} tokens — {total_toks/elapsed:.0f} tok/s")

            # Parse <think>...</think> to split reasoning from content
            rollout_results = []
            for text in raw_texts:
                if "</think>" in text:
                    parts = text.split("</think>", 1)
                    reasoning = parts[0].lstrip("<think>").strip()
                    content = parts[1].strip() if len(parts) > 1 else ""
                else:
                    reasoning = text.strip()
                    content = ""
                rollout_results.append({"reasoning": reasoning, "content": content})
            all_cot_rollouts.append(rollout_results)

        # Direct responses
        print(f"\nDirect responses: {len(direct_prompts)} prompts (skipping {len(problems) - len(direct_prompts)} no-ground-truth)...")
        t0 = time.time()
        raw_direct = run_generate(direct_prompts, direct_params)
        elapsed = time.time() - t0
        total_toks = count_tokens(raw_direct)
        print(f"  Done in {elapsed:.1f}s — {total_toks:,} tokens — {total_toks/elapsed:.0f} tok/s")

        empty_response = {"reasoning": "", "content": ""}
        direct_responses_map: dict[int, dict] = {}
        for j, idx in enumerate(direct_indices):
            direct_responses_map[idx] = {"reasoning": "", "content": raw_direct[j].strip()}
        direct_responses = [direct_responses_map.get(i, empty_response) for i in range(len(problems))]

        # Process results
        print("\nProcessing results...")
        results = process_results_openrouter(
            problems, all_cot_rollouts, direct_responses, tokenizer=tokenizer,
        )

        # Save with stats
        stats: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        saved = 0
        with open(output_path, "w") as f:
            for r in results:
                cat = r["category"] or "no_ground_truth"
                stats[cat] = stats.get(cat, 0) + 1
                source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1
                f.write(json.dumps(r) + "\n")
                saved += 1

        total_cot_chars = sum(len(r.get("cot_response", "")) for r in results)

        print(f"\n{'=' * 60}")
        print(f"FINAL STATS")
        print(f"{'=' * 60}")
        print(f"  Problems loaded: {len(problems)}")
        print(f"  Rollouts per problem: {n_rollouts}")
        print(f"  Total entries saved: {saved}")
        print(f"  Est. CoT tokens: {total_cot_chars // 4:,}")
        print(f"\n  By category:")
        for k, v in sorted(stats.items()):
            pct = v / saved * 100 if saved > 0 else 0
            print(f"    {k}: {v} ({pct:.1f}%)")
        print(f"\n  By source:")
        for k, v in sorted(source_counts.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
        print(f"\n  Output: {output_path}")


if __name__ == "__main__":
    main()

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
    """Load problems from MATH dataset."""
    for ds_name in ["hendrycks/competition_mathematics", "HuggingFaceH4/MATH-500"]:
        try:
            ds = load_dataset(ds_name, split="test")
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Could not load MATH dataset from any known source")

    problems = []
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
    return problems


def load_gsm8k_problems(n: int) -> list[dict]:
    """Load problems from GSM8K."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
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
    """Load problems from CommonsenseQA."""
    ds = load_dataset("tau/commonsense_qa", split="validation")
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
    """Load problems from AQUA-RAT (algebraic word problems)."""
    ds = load_dataset("deepmind/aqua_rat", "raw", split="test")
    problems = []
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


DATASET_LOADERS = {
    "math": lambda n, levels: load_math_problems(n, levels),
    "gsm8k": lambda n, levels: load_gsm8k_problems(n),
    "gpqa": lambda n, levels: load_gpqa_problems(n),
    "bbh": lambda n, levels: load_bbh_problems(n),
    "arc_challenge": lambda n, levels: load_arc_challenge_problems(n),
    "strategyqa": lambda n, levels: load_strategyqa_problems(n),
    "drop": lambda n, levels: load_drop_problems(n),
    "logiqa": lambda n, levels: load_logiqa_problems(n),
    "mmlu_pro": lambda n, levels: load_mmlu_pro_problems(n),
    "commonsenseqa": lambda n, levels: load_commonsenseqa_problems(n),
    "aqua_rat": lambda n, levels: load_aqua_rat_problems(n),
    "medqa": lambda n, levels: load_medqa_problems(n),
}

# Domain mapping: source name -> domain label (for Task 4: domain classification)
SOURCE_TO_DOMAIN = {
    "MATH": "math",
    "GSM8K": "math",
    "GPQA": "science",
    "BBH": "logic",
    "ARC": "science",
    "StrategyQA": "commonsense",
    "DROP": "reading",
    "LogiQA": "logic",
    "MMLU-Pro": "multi_domain",
    "CommonsenseQA": "commonsense",
    "AQUA-RAT": "math",
    "MedQA": "medical",
}


def load_all_problems(sources: list[str], n_per_source: int, levels: list[str] | None = None) -> list[dict]:
    """Load problems from specified sources."""
    problems = []
    for source in sources:
        loader = DATASET_LOADERS.get(source)
        if loader is None:
            raise ValueError(f"Unknown source: {source}. Available: {list(DATASET_LOADERS.keys())}")
        try:
            loaded = loader(n_per_source, levels)
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
    max_tokens: int = 2048,
    enable_thinking: bool = True,
) -> str:
    """Single OpenRouter API call for CoT generation."""
    global _completed_count, _failed_count
    async with semaphore:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        body = {
            "model": "qwen/qwen3-8b",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
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
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    _completed_count += 1
                    if _completed_count % 100 == 0:
                        print(f"  Progress: {_completed_count}/{_total_count} done, {_failed_count} failed")
                    return content
            except asyncio.TimeoutError:
                if attempt == 2:
                    _failed_count += 1
                    return ""
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 2:
                    _failed_count += 1
                    if _failed_count <= 10:
                        print(f"  OpenRouter error: {e}")
                    return ""
                await asyncio.sleep(2 ** attempt)
        return ""


async def openrouter_generate_batch(
    problems: list[dict],
    api_key: str,
    concurrency: int = 50,
    system_prompt: str | None = None,
    persona_name: str | None = None,
) -> tuple[list[str], list[str]]:
    """Generate CoT and direct responses for all problems via OpenRouter.

    Returns (cot_responses, direct_responses).
    """
    global _completed_count, _failed_count, _total_count
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        # CoT generation (gather preserves order)
        _completed_count = 0
        _failed_count = 0
        _total_count = len(problems)
        print(f"  Generating {len(problems)} CoT responses via OpenRouter...")
        cot_responses = list(await asyncio.gather(*[
            openrouter_generate_single(
                session, p["question"], api_key, semaphore,
                system_prompt=system_prompt,
                max_tokens=2048, enable_thinking=True,
            )
            for p in problems
        ]))
        print(f"  CoT done: {_completed_count} succeeded, {_failed_count} failed")

        # Direct (no-thinking) generation
        _completed_count = 0
        _failed_count = 0
        print(f"  Generating {len(problems)} direct responses via OpenRouter...")
        direct_responses = list(await asyncio.gather(*[
            openrouter_generate_single(
                session, p["question"], api_key, semaphore,
                system_prompt=system_prompt,
                max_tokens=200, enable_thinking=False,
            )
            for p in problems
        ]))
        print(f"  Direct done: {_completed_count} succeeded, {_failed_count} failed")

    return list(cot_responses), list(direct_responses)


# ============================================================
# Process Results
# ============================================================

def process_results_openrouter(
    problems: list[dict],
    cot_responses: list[str],
    direct_responses: list[str],
    tokenizer=None,
    persona: str | None = None,
) -> list[dict]:
    """Post-process OpenRouter responses into corpus entries.

    If tokenizer is provided, computes boundary_positions (needed for
    sentence-structured training tasks). No GPU needed — just tokenizer.
    """
    results = []
    for i, (problem, cot_response, direct_response) in enumerate(
        zip(problems, cot_responses, direct_responses)
    ):
        if not cot_response:
            continue

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

        # Compute boundary positions if tokenizer available
        boundary_positions = []
        if tokenizer is not None:
            messages = [{"role": "user", "content": question}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            full_text = formatted + cot_response
            boundary_positions = find_sentence_boundary_positions(
                tokenizer, full_text, sentences,
            )

        domain = SOURCE_TO_DOMAIN.get(problem["source"], "unknown")

        entry = {
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

        messages = [{"role": "user", "content": question}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        full_text = formatted + cot_response
        boundary_positions = find_sentence_boundary_positions(tokenizer, full_text, sentences)

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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate CoT corpus")
    parser.add_argument("--sources", nargs="+",
                        default=["math", "gsm8k", "gpqa", "bbh", "arc_challenge",
                                 "strategyqa", "drop", "logiqa", "mmlu_pro",
                                 "commonsenseqa", "aqua_rat", "medqa"])
    parser.add_argument("--n-problems", type=int, default=500, help="Problems per source")
    parser.add_argument("--levels", nargs="*", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", default="data/cot_corpus/corpus.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--keep-all", action="store_true", default=True,
                        help="Keep all CoTs, not just load-bearing (default: True)")
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load problems
    problems = load_all_problems(args.sources, args.n_problems, args.levels)

    if args.openrouter:
        import os
        api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY env var or pass --openrouter-key")

        # Try loading tokenizer for boundary position computation (no GPU needed).
        # Falls back gracefully — boundary_positions will be computed on GPU later.
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer for {args.model}...")
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            print("Tokenizer loaded — will compute boundary_positions")
        except Exception as e:
            print(f"Tokenizer not available ({e}) — boundary_positions deferred to GPU")

        if args.personas:
            # Persona mode: generate for each persona
            persona_names = args.persona_list or list(PERSONAS.keys())
            all_results = []

            for persona_name in persona_names:
                system_prompt = PERSONAS.get(persona_name)
                print(f"\n{'=' * 60}")
                print(f"Generating for persona: {persona_name}")
                print(f"  System prompt: {system_prompt or '(none)'}")
                print(f"{'=' * 60}")

                cot_responses, direct_responses = asyncio.run(
                    openrouter_generate_batch(
                        problems, api_key,
                        concurrency=args.concurrency,
                        system_prompt=system_prompt,
                        persona_name=persona_name,
                    )
                )

                results = process_results_openrouter(
                    problems, cot_responses, direct_responses,
                    tokenizer=tokenizer, persona=persona_name,
                )
                all_results.extend(results)
                print(f"  {persona_name}: {len(results)} entries")

            # Save all persona results
            saved = 0
            with open(output_path, "w") as f:
                for r in all_results:
                    f.write(json.dumps(r) + "\n")
                    saved += 1

            print(f"\nTotal persona entries: {saved}")
            print(f"Output: {output_path}")

        else:
            # Standard OpenRouter mode (no personas)
            cot_responses, direct_responses = asyncio.run(
                openrouter_generate_batch(
                    problems, api_key,
                    concurrency=args.concurrency,
                )
            )

            results = process_results_openrouter(
                problems, cot_responses, direct_responses, tokenizer=tokenizer,
            )

            stats = {"load_bearing": 0, "both_correct": 0, "both_wrong": 0, "cot_hurt": 0}
            saved = 0
            with open(output_path, "w") as f:
                for r in results:
                    stats[r["category"]] += 1
                    if r["category"] == "load_bearing" or args.keep_all:
                        f.write(json.dumps(r) + "\n")
                        saved += 1

            print(f"\n{'=' * 60}")
            print(f"FINAL STATS")
            print(f"{'=' * 60}")
            total = sum(stats.values())
            for k, v in stats.items():
                pct = v / total * 100 if total > 0 else 0
                print(f"  {k}: {v} ({pct:.1f}%)")
            print(f"  Total saved: {saved}")
            print(f"  Output: {output_path}")

    else:
        # Local GPU mode (original)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {args.model} (bf16, no AO)...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
        )

        print(f"\nGenerating CoTs (batch_size={args.batch_size})...")
        questions = [p["question"] for p in problems]

        all_cot = []
        all_direct = []
        for i in tqdm(range(0, len(questions), args.batch_size), desc="Batches"):
            batch_q = questions[i:i + args.batch_size]

            cot_prompts = []
            direct_prompts = []
            for q in batch_q:
                msgs = [{"role": "user", "content": q}]
                cot_prompts.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                ))
                direct_prompts.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                ))

            cot_responses = batch_generate(model, tokenizer, cot_prompts, max_new_tokens=1024)
            all_cot.extend(cot_responses)

            direct_responses = batch_generate(model, tokenizer, direct_prompts, max_new_tokens=100)
            all_direct.extend(direct_responses)

        print("\nProcessing results...")
        results = process_results(problems, all_cot, all_direct, tokenizer)

        stats = {"load_bearing": 0, "both_correct": 0, "both_wrong": 0, "cot_hurt": 0}
        saved = 0
        with open(output_path, "w") as f:
            for r in results:
                stats[r["category"]] += 1
                if r["category"] == "load_bearing" or args.keep_all:
                    f.write(json.dumps(r) + "\n")
                    saved += 1

        print(f"\n{'=' * 60}")
        print(f"FINAL STATS")
        print(f"{'=' * 60}")
        total = sum(stats.values())
        for k, v in stats.items():
            pct = v / total * 100 if total > 0 else 0
            print(f"  {k}: {v} ({pct:.1f}%)")
        print(f"  Skipped (< 2 sentences): {len(problems) - total}")
        print(f"  Total saved: {saved}")
        print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()

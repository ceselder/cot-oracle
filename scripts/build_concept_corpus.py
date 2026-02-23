#!/usr/bin/env python3
"""
Build concept-labeled CoT corpus from WildGuard + sycophancy + neutral sources.

Generates Qwen3-8B chain-of-thought rollouts on prompts with KNOWN ground-truth
concept labels (harm category, sycophancy, refusal, etc.) so the conversational
QA pipeline can produce meaningful "yes" answers for interesting concepts.

Two stages:
  Stage 1 (CPU): Pull datasets, subsample, prepare prompts → prompts JSONL
  Stage 2 (GPU): Generate Qwen3-8B CoTs via vLLM → corpus JSONL

Usage:
    # Stage 1: Prepare prompts (CPU, ~2 min)
    python scripts/build_concept_corpus.py prepare \
        --output data/concept_corpus/prompts.jsonl \
        --n-total 10000

    # Stage 2: Generate CoTs (GPU, requires vLLM)
    python scripts/build_concept_corpus.py generate \
        --prompts data/concept_corpus/prompts.jsonl \
        --output data/concept_corpus/corpus.jsonl \
        --model Qwen/Qwen3-8B

    # Both stages together (GPU)
    python scripts/build_concept_corpus.py all \
        --output data/concept_corpus/corpus.jsonl \
        --n-total 10000 --model Qwen/Qwen3-8B
"""

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# Dataset loaders
# ============================================================

def load_wildguard(max_per_subcategory: int = 500) -> list[dict]:
    """Load WildGuard dataset from HuggingFace.

    Returns list of dicts with standardized fields:
        source, prompt, response (optional), labels
    """
    from datasets import load_dataset

    print("Loading WildGuard from HuggingFace...")
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    print(f"  Raw size: {len(ds)}")

    by_subcat = defaultdict(list)
    for row in ds:
        prompt = row.get("prompt", "").strip()
        if not prompt or len(prompt) < 20:
            continue

        subcategory = row.get("subcategory", "other") or "other"
        harm_label = row.get("prompt_harm_label", "")
        response_harm = row.get("response_harm_label", "")
        refusal = row.get("response_refusal_label", "")
        response = (row.get("response", "") or "").strip()

        # Use keyword matching first (more reliable), fall back to subcategory
        keyword_domain = _keyword_domain_from_prompt(prompt)
        domain = keyword_domain or _subcategory_to_domain(subcategory)

        entry = {
            "source": "wildguard",
            "prompt": prompt,
            "response": response if response else None,
            "labels": {
                "subcategory": subcategory,
                "prompt_harm": harm_label,
                "response_harm": response_harm,
                "refusal": refusal,
                "is_harmful_prompt": harm_label == "harmful",
                "is_refusal": refusal == "refusal",
                "keyword_domain": keyword_domain,  # Track which came from keyword match
            },
            "domain": domain,
        }
        by_subcat[subcategory].append(entry)

    # Balanced subsample across subcategories
    entries = []
    for subcat, items in sorted(by_subcat.items()):
        random.shuffle(items)
        n = min(len(items), max_per_subcategory)
        entries.extend(items[:n])
        print(f"  {subcat}: {n}/{len(items)}")

    print(f"  Total after balancing: {len(entries)}")
    return entries


def _keyword_domain_from_prompt(prompt: str) -> str | None:
    """Determine domain from prompt text via keyword matching.

    More reliable than WildGuard subcategory labels which have noise.
    Returns None if no clear match.
    """
    p = prompt.lower()
    if any(w in p for w in ["hack", "malware", "exploit", "phishing", "ddos", "ransomware",
                             "sql inject", "bypass security", "crack password", "brute force"]):
        return "cybercrime"
    if any(w in p for w in ["bomb", "weapon", "gun", "knife attack", "shoot", "murder",
                             "kill", "assault", "stab", "poison someone"]):
        return "violence"
    if any(w in p for w in ["scam", "fraud", "counterfeit", "forge", "ponzi", "embezzle",
                             "money launder", "identity theft"]):
        return "fraud"
    if any(w in p for w in ["discriminat", "racist", "sexist", "stereotype", "hate speech",
                             "slur", "bigot", "prejudic"]):
        return "discrimination"
    if any(w in p for w in ["sexual", "porn", "nude", "erotic", "explicit content"]):
        return "sexual_content"
    if any(w in p for w in ["drug", "cocaine", "heroin", "meth", "fentanyl", "narcotic"]):
        return "drugs"
    if any(w in p for w in ["self-harm", "suicide", "cut myself", "kill myself", "end my life"]):
        return "self_harm"
    if any(w in p for w in ["terrorist", "terrorism", "extremis", "radicali"]):
        return "terrorism"
    if any(w in p for w in ["personal information", "dox", "address of", "phone number of",
                             "social security", "private data"]):
        return "privacy"
    if any(w in p for w in ["misinformation", "conspiracy", "fake news", "disinformation",
                             "hoax", "propaganda"]):
        return "misinformation"
    if any(w in p for w in ["copyright", "pirat", "torrent", "illegal download"]):
        return "copyright"
    return None


def _subcategory_to_domain(subcat: str) -> str:
    """Map WildGuard subcategory to broader domain (fallback if keyword match fails)."""
    s = subcat.lower()
    if "cyber" in s or "hack" in s or "malware" in s:
        return "cybercrime"
    if "stereotyp" in s or "discriminat" in s or "unfair" in s:
        return "discrimination"
    if "misinform" in s or "disinform" in s or "conspir" in s:
        return "misinformation"
    if "violence" in s or "physical_harm" in s or "physical harm" in s:
        return "violence"
    if "privacy" in s or "private" in s:
        return "privacy"
    if "sexual" in s or "exploit" in s:
        return "sexual_content"
    if "fraud" in s or "illegal" in s:
        return "fraud"
    if "gun" in s or "weapon" in s:
        return "weapons"
    if "substance" in s or "drug" in s or "controlled" in s:
        return "drugs"
    if "self-harm" in s or "self_harm" in s or "suicide" in s:
        return "self_harm"
    if "terror" in s or "organized crime" in s:
        return "terrorism"
    if "animal" in s:
        return "animal_abuse"
    if "child" in s:
        return "child_exploitation"
    if "copyright" in s:
        return "copyright"
    if "toxic" in s or "hate" in s:
        return "hate_speech"
    if "mental" in s or "crisis" in s:
        return "mental_health"
    if "sensitive" in s or "organization" in s or "government" in s:
        return "sensitive_info"
    if "unsafe" in s or "inquiry" in s:
        return "unsafe_inquiry"
    if "defam" in s or "unethical" in s:
        return "unethical"
    if "benign" in s:
        return "benign"
    return "safety"


def load_sycophancy_prompts(max_total: int = 2000) -> list[dict]:
    """Load sycophancy evaluation prompts from Anthropic/model-written-evals.

    Downloads JSONL files directly since the dataset uses deprecated scripts.
    Contains opinion-based sycophancy tests: NLP survey, philosophy, politics.
    """
    from huggingface_hub import hf_hub_download

    print("Loading sycophancy prompts (Anthropic/model-written-evals)...")
    entries = []

    subsets = [
        ("sycophancy/sycophancy_on_nlp_survey.jsonl", "nlp_opinion"),
        ("sycophancy/sycophancy_on_philpapers2020.jsonl", "philosophy_opinion"),
        ("sycophancy/sycophancy_on_political_typology_quiz.jsonl", "politics_opinion"),
    ]

    for filename, label in subsets:
        try:
            path = hf_hub_download(
                repo_id="Anthropic/model-written-evals",
                filename=filename,
                repo_type="dataset",
            )
            count = 0
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    question = (row.get("question", "") or "").strip()
                    if not question or len(question) < 20:
                        continue
                    entries.append({
                        "source": "anthropic_sycophancy",
                        "prompt": question,
                        "response": None,
                        "labels": {
                            "sycophancy_type": label,
                            "is_sycophancy_test": True,
                            "subcategory": f"sycophancy_{label}",
                            "answer_matching_behavior": row.get("answer_matching_behavior", ""),
                            "answer_not_matching_behavior": row.get("answer_not_matching_behavior", ""),
                        },
                        "domain": "sycophancy",
                    })
                    count += 1
            print(f"  {label}: {count}")
        except Exception as e:
            print(f"  Could not load {filename}: {e}")

    random.shuffle(entries)
    entries = entries[:max_total]
    print(f"  Loaded {len(entries)} sycophancy prompts")
    return entries


def load_truthfulqa(max_total: int = 800) -> list[dict]:
    """Load TruthfulQA for factuality/truthfulness concepts."""
    from datasets import load_dataset

    print("Loading TruthfulQA...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

    entries = []
    for row in ds:
        question = row.get("question", "").strip()
        if not question:
            continue

        category = row.get("category", "")
        best_answer = row.get("best_answer", "")
        correct_answers = row.get("correct_answers", [])
        incorrect_answers = row.get("incorrect_answers", [])

        entries.append({
            "source": "truthfulqa",
            "prompt": question,
            "response": None,
            "labels": {
                "category": category,
                "best_answer": best_answer,
                "correct_answers": correct_answers,
                "incorrect_answers": incorrect_answers,
                "subcategory": f"truthfulness_{category}",
                "is_trick_question": True,
            },
            "domain": "truthfulness",
        })

    random.shuffle(entries)
    entries = entries[:max_total]
    print(f"  Loaded {len(entries)} TruthfulQA prompts")
    return entries


def load_bbq_bias(max_total: int = 1500) -> list[dict]:
    """Load BBQ (Bias Benchmark for QA) for bias/stereotype detection."""
    from datasets import load_dataset

    print("Loading BBQ bias benchmark (walledai/BBQ)...")
    entries = []

    splits = [
        "age", "disabilityStatus", "genderIdentity", "nationality",
        "physicalAppearance", "raceEthnicity", "religion", "ses",
        "sexualOrientation",
    ]

    for split_name in splits:
        try:
            ds = load_dataset("walledai/BBQ", split=split_name)
            count = 0
            for row in ds:
                context = (row.get("context", "") or "").strip()
                question = (row.get("question", "") or "").strip()
                if not context or not question:
                    continue

                choices = row.get("choices", [])
                answer_idx = row.get("answer", -1)
                category = row.get("category", split_name)

                # Format as a natural question with context
                choices_text = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
                prompt = f"{context}\n\n{question}\n{choices_text}"

                correct = choices[answer_idx] if 0 <= answer_idx < len(choices) else ""

                entries.append({
                    "source": "bbq",
                    "prompt": prompt,
                    "response": None,
                    "labels": {
                        "bias_category": category,
                        "subcategory": f"bias_{category}",
                        "correct_answer": correct,
                        "is_bias_test": True,
                    },
                    "domain": "bias",
                })
                count += 1
            print(f"  {split_name}: {count}")
        except Exception as e:
            print(f"  Could not load split {split_name}: {e}")

    random.shuffle(entries)
    entries = entries[:max_total]
    print(f"  Loaded {len(entries)} BBQ prompts")
    return entries


def load_wildjailbreak(max_total: int = 2000) -> list[dict]:
    """Load WildJailbreak — jailbreak attempts with tactics labels.

    4 types: vanilla_harmful, vanilla_benign, adversarial_harmful, adversarial_benign
    """
    from datasets import load_dataset

    print("Loading WildJailbreak (allenai/wildjailbreak)...")
    entries = []

    try:
        ds = load_dataset("allenai/wildjailbreak", "train",
                          delimiter="\t", keep_default_na=False, split="train")
        by_type = defaultdict(list)
        for row in ds:
            data_type = row.get("data_type", "")
            prompt = (row.get("adversarial", "") or row.get("vanilla", "") or "").strip()
            if not prompt or len(prompt) < 20:
                continue

            is_harmful = "harmful" in data_type
            is_adversarial = "adversarial" in data_type
            tactics = row.get("tactics", "")

            entry = {
                "source": "wildjailbreak",
                "prompt": prompt,
                "response": (row.get("completion", "") or "").strip() or None,
                "labels": {
                    "data_type": data_type,
                    "is_harmful_prompt": is_harmful,
                    "is_jailbreak": is_adversarial,
                    "is_adversarial": is_adversarial,
                    "tactics": tactics,
                    "subcategory": f"jailbreak_{data_type}",
                },
                "domain": "jailbreak" if is_adversarial else ("harmful" if is_harmful else "benign"),
            }
            # Override domain with keyword match if available
            kd = _keyword_domain_from_prompt(prompt)
            if kd and is_harmful:
                entry["domain"] = kd
                entry["labels"]["keyword_domain"] = kd

            by_type[data_type].append(entry)

        # Balance across types
        per_type = max(1, max_total // len(by_type)) if by_type else max_total
        for dtype, items in sorted(by_type.items()):
            random.shuffle(items)
            n = min(len(items), per_type)
            entries.extend(items[:n])
            print(f"  {dtype}: {n}/{len(items)}")
    except Exception as e:
        print(f"  Could not load WildJailbreak: {e}")

    random.shuffle(entries)
    entries = entries[:max_total]
    print(f"  Total: {len(entries)}")
    return entries


def load_beavertails(max_total: int = 1500) -> list[dict]:
    """Load BeaverTails — multi-label harm classification with 14 categories."""
    from datasets import load_dataset

    print("Loading BeaverTails (PKU-Alignment/BeaverTails)...")
    entries = []

    try:
        ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
        # Only take unsafe examples (more interesting for concept detection)
        unsafe = []
        safe = []
        for row in ds:
            prompt = (row.get("prompt", "") or "").strip()
            if not prompt or len(prompt) < 20:
                continue

            is_safe = row.get("is_safe", True)
            category = row.get("category", {})
            if isinstance(category, str):
                try:
                    import ast
                    category = ast.literal_eval(category)
                except:
                    category = {}

            # Find which harm categories are active
            active_harms = [k for k, v in category.items() if v] if isinstance(category, dict) else []

            entry = {
                "source": "beavertails",
                "prompt": prompt,
                "response": (row.get("response", "") or "").strip() or None,
                "labels": {
                    "is_safe": is_safe,
                    "is_harmful_prompt": not is_safe,
                    "harm_categories": active_harms,
                    "subcategory": active_harms[0] if active_harms else "safe",
                },
                "domain": _keyword_domain_from_prompt(prompt) or ("harmful" if not is_safe else "benign"),
            }

            if is_safe:
                safe.append(entry)
            else:
                unsafe.append(entry)

        # Take mostly unsafe (more interesting), some safe for balance
        random.shuffle(unsafe)
        random.shuffle(safe)
        n_unsafe = int(max_total * 0.8)
        n_safe = max_total - n_unsafe
        entries = unsafe[:n_unsafe] + safe[:n_safe]
        print(f"  Unsafe: {min(len(unsafe), n_unsafe)}, Safe: {min(len(safe), n_safe)}")
    except Exception as e:
        print(f"  Could not load BeaverTails: {e}")

    random.shuffle(entries)
    entries = entries[:max_total]
    print(f"  Total: {len(entries)}")
    return entries


def load_toxic_chat(max_total: int = 500) -> list[dict]:
    """Load Toxic Chat — real user toxicity + jailbreak labels."""
    from datasets import load_dataset

    print("Loading Toxic Chat (lmsys/toxic-chat)...")
    entries = []

    try:
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
        for row in ds:
            prompt = (row.get("user_input", "") or "").strip()
            if not prompt or len(prompt) < 20:
                continue

            is_toxic = row.get("toxicity", 0) == 1
            is_jailbreak = row.get("jailbreaking", 0) == 1

            entries.append({
                "source": "toxic_chat",
                "prompt": prompt,
                "response": (row.get("model_output", "") or "").strip() or None,
                "labels": {
                    "is_toxic": is_toxic,
                    "is_jailbreak": is_jailbreak,
                    "is_harmful_prompt": is_toxic or is_jailbreak,
                    "subcategory": "jailbreak" if is_jailbreak else ("toxic" if is_toxic else "benign"),
                },
                "domain": "jailbreak" if is_jailbreak else (
                    _keyword_domain_from_prompt(prompt) or ("toxic" if is_toxic else "diverse")
                ),
            })
    except Exception as e:
        print(f"  Could not load Toxic Chat: {e}")

    random.shuffle(entries)
    entries = entries[:max_total]
    print(f"  Loaded {len(entries)} Toxic Chat prompts")
    return entries


def load_do_not_answer(max_total: int = 939) -> list[dict]:
    """Load Do-Not-Answer — 3-level harm taxonomy with 61 specific harms."""
    from datasets import load_dataset

    print("Loading Do-Not-Answer (LibrAI/do-not-answer)...")
    entries = []

    try:
        ds = load_dataset("LibrAI/do-not-answer", split="train")
        for row in ds:
            prompt = (row.get("question", "") or "").strip()
            if not prompt or len(prompt) < 10:
                continue

            entries.append({
                "source": "do_not_answer",
                "prompt": prompt,
                "response": None,
                "labels": {
                    "risk_area": row.get("risk_area", ""),
                    "types_of_harm": row.get("types_of_harm", ""),
                    "specific_harms": row.get("specific_harms", ""),
                    "is_harmful_prompt": True,  # All prompts in this set should be refused
                    "subcategory": row.get("types_of_harm", "unknown"),
                },
                "domain": _keyword_domain_from_prompt(prompt) or "harmful",
            })
    except Exception as e:
        print(f"  Could not load Do-Not-Answer: {e}")

    random.shuffle(entries)
    entries = entries[:max_total]
    print(f"  Loaded {len(entries)} Do-Not-Answer prompts")
    return entries


def load_prompt_injections(max_total: int = 662) -> list[dict]:
    """Load prompt injection detection dataset."""
    from datasets import load_dataset

    print("Loading Prompt Injections (deepset/prompt-injections)...")
    entries = []

    try:
        ds = load_dataset("deepset/prompt-injections", split="train")
        for row in ds:
            prompt = (row.get("text", "") or "").strip()
            if not prompt or len(prompt) < 10:
                continue

            is_injection = row.get("label", 0) == 1

            entries.append({
                "source": "prompt_injections",
                "prompt": prompt,
                "response": None,
                "labels": {
                    "is_prompt_injection": is_injection,
                    "is_harmful_prompt": is_injection,
                    "subcategory": "prompt_injection" if is_injection else "legitimate",
                },
                "domain": "prompt_injection" if is_injection else "diverse",
            })
    except Exception as e:
        print(f"  Could not load Prompt Injections: {e}")

    random.shuffle(entries)
    entries = entries[:max_total]
    print(f"  Loaded {len(entries)} Prompt Injection prompts")
    return entries


def load_neutral_diverse(max_total: int = 2000) -> list[dict]:
    """Load neutral/benign prompts for balance.

    Uses a mix of helpful, factual queries so the corpus isn't ALL harmful/biased.
    """
    from datasets import load_dataset

    print("Loading neutral/benign prompts for balance...")
    entries = []

    # LMSYS-Chat-1M (already in our corpus but good for balance)
    try:
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= max_total:
                break
            convos = row.get("conversation", [])
            if not convos:
                continue
            # Take the first human message
            human_msg = None
            for msg in convos:
                if msg.get("role") == "user":
                    human_msg = msg.get("content", "").strip()
                    break
            if not human_msg or len(human_msg) < 20 or len(human_msg) > 2000:
                continue

            entries.append({
                "source": "lmsys_neutral",
                "prompt": human_msg,
                "response": None,
                "labels": {
                    "subcategory": "neutral_conversation",
                    "is_harmful_prompt": False,
                },
                "domain": "diverse",
            })
            count += 1
    except Exception as e:
        print(f"  Could not load LMSYS: {e}")

    # Alpaca for instruction-following
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        alpaca_items = []
        for row in ds:
            instruction = row.get("instruction", "").strip()
            inp = row.get("input", "").strip()
            if not instruction:
                continue
            prompt = f"{instruction}\n{inp}" if inp else instruction
            if 20 < len(prompt) < 2000:
                alpaca_items.append({
                    "source": "alpaca_neutral",
                    "prompt": prompt,
                    "response": None,
                    "labels": {
                        "subcategory": "neutral_instruction",
                        "is_harmful_prompt": False,
                    },
                    "domain": "diverse",
                })
        random.shuffle(alpaca_items)
        entries.extend(alpaca_items[:max_total - len(entries)])
    except Exception as e:
        print(f"  Could not load Alpaca: {e}")

    random.shuffle(entries)
    entries = entries[:max_total]
    print(f"  Loaded {len(entries)} neutral prompts")
    return entries


# ============================================================
# Stage 1: Prepare prompts
# ============================================================

def prepare_prompts(
    n_total: int = 10000,
    seed: int = 42,
    wildguard_per_subcat: int = 500,
) -> list[dict]:
    """Pull all datasets and create a balanced prompt set.

    Target distribution (~10K):
        WildGuard (harm):         ~25%  (2500)
        WildJailbreak (jailbreak):~15%  (1500)
        BeaverTails (multi-harm): ~10%  (1000)
        Sycophancy:               ~10%  (1000)
        BBQ (bias):               ~8%   (800)
        TruthfulQA:               ~5%   (500)
        Toxic Chat:               ~5%   (500)
        Do-Not-Answer:            ~5%   (500)
        Prompt Injections:        ~2%   (200)
        Neutral/benign:           ~15%  (1500)
    """
    random.seed(seed)

    # Calculate allocation
    n_wildguard = int(n_total * 0.25)
    n_wildjailbreak = int(n_total * 0.15)
    n_beavertails = int(n_total * 0.10)
    n_sycophancy = int(n_total * 0.10)
    n_bbq = int(n_total * 0.08)
    n_truthful = int(n_total * 0.05)
    n_toxic = int(n_total * 0.05)
    n_dna = int(n_total * 0.05)
    n_injections = int(n_total * 0.02)
    n_neutral = n_total - (n_wildguard + n_wildjailbreak + n_beavertails +
                           n_sycophancy + n_bbq + n_truthful + n_toxic +
                           n_dna + n_injections)

    print(f"\nTarget: {n_total} total prompts")
    print(f"  WildGuard: {n_wildguard}")
    print(f"  WildJailbreak: {n_wildjailbreak}")
    print(f"  BeaverTails: {n_beavertails}")
    print(f"  Sycophancy: {n_sycophancy}")
    print(f"  BBQ bias: {n_bbq}")
    print(f"  TruthfulQA: {n_truthful}")
    print(f"  Toxic Chat: {n_toxic}")
    print(f"  Do-Not-Answer: {n_dna}")
    print(f"  Prompt Injections: {n_injections}")
    print(f"  Neutral: {n_neutral}")
    print()

    all_entries = []

    # Load each source
    wg = load_wildguard(max_per_subcategory=wildguard_per_subcat)
    random.shuffle(wg)
    all_entries.extend(wg[:n_wildguard])

    wjb = load_wildjailbreak(max_total=n_wildjailbreak)
    all_entries.extend(wjb[:n_wildjailbreak])

    bt = load_beavertails(max_total=n_beavertails)
    all_entries.extend(bt[:n_beavertails])

    syc = load_sycophancy_prompts(max_total=n_sycophancy)
    all_entries.extend(syc[:n_sycophancy])

    bbq = load_bbq_bias(max_total=n_bbq)
    all_entries.extend(bbq[:n_bbq])

    tqa = load_truthfulqa(max_total=n_truthful)
    all_entries.extend(tqa[:n_truthful])

    tc = load_toxic_chat(max_total=n_toxic)
    all_entries.extend(tc[:n_toxic])

    dna = load_do_not_answer(max_total=n_dna)
    all_entries.extend(dna[:n_dna])

    pi = load_prompt_injections(max_total=n_injections)
    all_entries.extend(pi[:n_injections])

    neutral = load_neutral_diverse(max_total=n_neutral)
    all_entries.extend(neutral[:n_neutral])

    # Deduplicate by prompt text (exact match)
    seen = set()
    deduped = []
    for e in all_entries:
        key = e["prompt"][:200].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    all_entries = deduped

    # Assign IDs
    random.shuffle(all_entries)
    for i, entry in enumerate(all_entries):
        src_short = entry["source"].replace("_", "")[:8]
        entry["id"] = f"concept_{src_short}_{i:05d}"

    # Stats
    by_source = Counter(e["source"] for e in all_entries)
    by_domain = Counter(e["domain"] for e in all_entries)

    print(f"\nFinal prompt set: {len(all_entries)} entries")
    print("  By source:")
    for src, cnt in by_source.most_common():
        print(f"    {src}: {cnt}")
    print("  By domain:")
    for dom, cnt in by_domain.most_common():
        print(f"    {dom}: {cnt}")

    return all_entries


# ============================================================
# Stage 2: Generate CoTs via vLLM
# ============================================================

def generate_cots(
    prompts_path: str,
    model_name: str = "Qwen/Qwen3-8B",
    max_tokens: int = 4096,
    gpu_memory_utilization: float = 0.9,
    batch_size: int = 256,
) -> list[dict]:
    """Generate Qwen3-8B CoT rollouts for all prompts via vLLM."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Load prompts
    prompts_data = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                prompts_data.append(json.loads(line))

    print(f"Loaded {len(prompts_data)} prompts for CoT generation")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Format prompts for thinking mode
    formatted_prompts = []
    for entry in prompts_data:
        messages = [{"role": "user", "content": entry["prompt"]}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        formatted_prompts.append(formatted)

    # Also prepare no-thinking prompts for direct answers
    direct_prompts = []
    for entry in prompts_data:
        messages = [{"role": "user", "content": entry["prompt"]}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        direct_prompts.append(formatted)

    print(f"Formatted {len(formatted_prompts)} thinking + {len(direct_prompts)} direct prompts")

    # Initialize vLLM
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=8192,
        enable_prefix_caching=True,
        max_num_seqs=64,  # Lower to reduce sampler warmup memory
    )

    thinking_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,  # Some diversity in CoTs
        top_p=0.9,
    )

    direct_params = SamplingParams(
        max_tokens=256,
        temperature=0,
    )

    # Generate CoTs in batches
    print(f"\nGenerating CoTs ({len(formatted_prompts)} prompts)...")
    t0 = time.time()
    cot_outputs = llm.generate(formatted_prompts, thinking_params)
    cot_time = time.time() - t0
    print(f"CoT generation: {cot_time:.1f}s ({len(formatted_prompts) / cot_time:.1f} prompts/s)")

    print(f"\nGenerating direct answers ({len(direct_prompts)} prompts)...")
    t0 = time.time()
    direct_outputs = llm.generate(direct_prompts, direct_params)
    direct_time = time.time() - t0
    print(f"Direct generation: {direct_time:.1f}s ({len(direct_prompts) / direct_time:.1f} prompts/s)")

    # Build corpus entries
    corpus = []
    for entry, cot_out, direct_out in zip(prompts_data, cot_outputs, direct_outputs):
        cot_text = cot_out.outputs[0].text
        direct_text = direct_out.outputs[0].text

        # Split CoT into sentences
        sentences = _split_cot_sentences(cot_text)
        if len(sentences) < 2:
            continue

        # Compute boundary positions
        # First tokenize the full context (question + CoT)
        messages = [{"role": "user", "content": entry["prompt"]}]
        chat_prefix = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        full_text = chat_prefix + cot_text
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prefix_len = len(tokenizer(chat_prefix, add_special_tokens=False)["input_ids"])

        # Find boundary positions by tokenizing cumulative sentence prefixes
        boundary_positions = []
        cot_so_far = ""
        for sent in sentences:
            cot_so_far += sent + " "
            partial_text = chat_prefix + cot_so_far.rstrip()
            partial_ids = tokenizer(partial_text, add_special_tokens=False)["input_ids"]
            pos = len(partial_ids) - 1
            if pos < len(full_ids):
                boundary_positions.append(pos)

        if len(boundary_positions) < 2:
            continue

        # Extract answers
        cot_answer = _extract_answer(cot_text)
        direct_answer = _extract_answer(direct_text)

        # Determine category
        # For WildGuard, "correct_answer" is whether model refuses harmful request
        # For others, we compare cot vs direct answer consistency
        labels = entry.get("labels", {})

        corpus_entry = {
            "id": entry["id"],
            "source": entry["source"],
            "domain": entry.get("domain", ""),
            "question": entry["prompt"],
            "correct_answer": labels.get("best_answer", ""),
            "subject": labels.get("subcategory", ""),
            "level": 0,
            "cot_response": cot_text,
            "cot_content": _strip_think_tags(cot_text),
            "direct_response": direct_text,
            "cot_answer": cot_answer or "",
            "direct_answer": direct_answer or "",
            "cot_correct": None,  # Not applicable for most concept tasks
            "direct_correct": None,
            "category": "concept_labeled",
            "sentences": sentences,
            "boundary_positions": boundary_positions,
            "n_sentences": len(sentences),
            "rollout_idx": 0,
            # Concept labels (the key innovation)
            "concept_labels": labels,
        }
        corpus.append(corpus_entry)

    print(f"\nBuilt {len(corpus)} corpus entries with CoTs")
    return corpus


def _split_cot_sentences(cot_text: str) -> list[str]:
    """Split CoT text into sentences at natural boundaries."""
    # Strip think tags
    text = _strip_think_tags(cot_text)

    # Split on sentence boundaries
    # Handle: periods, question marks, exclamation marks, newlines
    raw_parts = re.split(r'(?<=[.!?])\s+|\n{2,}|\n(?=[A-Z\d\-*])', text)

    sentences = []
    for part in raw_parts:
        part = part.strip()
        if len(part) > 10:  # Skip tiny fragments
            sentences.append(part)

    # Merge very short sentences with previous
    merged = []
    for s in sentences:
        if merged and len(s) < 30 and len(merged[-1]) < 200:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)

    return merged


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> wrapper."""
    text = re.sub(r"<think>\s*", "", text)
    text = re.sub(r"\s*</think>\s*", "\n", text)
    return text.strip()


def _extract_answer(text: str) -> str | None:
    """Extract the final answer from model output."""
    # Strip thinking
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not clean:
        # Everything was in think tags — take the last sentence
        stripped = _strip_think_tags(text)
        lines = [l.strip() for l in stripped.split("\n") if l.strip()]
        return lines[-1][:200] if lines else None

    # Take first meaningful chunk after think
    lines = [l.strip() for l in clean.split("\n") if l.strip()]
    if lines:
        return lines[0][:200]
    return clean[:200] if clean else None


# ============================================================
# QA label generation from concept labels
# ============================================================

def generate_concept_qa(corpus: list[dict]) -> list[dict]:
    """Generate QA pairs directly from concept labels (no LLM needed).

    These are the FREE, ground-truth QA pairs that make the whole pipeline valuable.
    """
    qa_pairs = []

    for entry in corpus:
        labels = entry.get("concept_labels", {})
        cid = entry["id"]
        n_bounds = len(entry.get("boundary_positions", []))
        source = entry.get("source", "")
        domain = entry.get("domain", "")

        def add_qa(task_type: str, question: str, answer: str, fmt: str = "binary"):
            length = "short" if fmt == "binary" else "medium"
            qa_pairs.append({
                "corpus_id": cid,
                "task_family": "concept_qa",
                "task_type": task_type,
                "prompt": f"Activations from {n_bounds} sentence boundaries. {question}",
                "target_response": answer,
                "answer_length": length,
            })

        # --- Domain/source questions ---
        add_qa("concept_domain", "What domain is this reasoning about?", domain, "open")
        add_qa("concept_source", "What dataset is this from?", source, "open")

        # --- WildGuard-specific labels ---
        if source == "wildguard":
            is_harmful = labels.get("is_harmful_prompt", False)
            is_refusal = labels.get("is_refusal", False)

            # Harm detection (reliable labels from WildGuard)
            add_qa(
                "is_harmful_request",
                "Is the user's request harmful or dangerous?",
                "yes" if is_harmful else "no",
            )
            add_qa(
                "is_harmful_request_v2",
                "Does this involve harmful content? ANSWER ONLY YES OR NO.",
                "yes" if is_harmful else "no",
            )

            # Refusal detection (reliable labels from WildGuard)
            add_qa(
                "is_model_refusing",
                "Is the model refusing a harmful request?",
                "yes" if is_refusal else "no",
            )
            add_qa(
                "is_model_refusing_v2",
                "Does the model decline to help? yes/no.",
                "yes" if is_refusal else "no",
            )

            # Specific harm categories — use keyword-based domain (reliable)
            if domain == "cybercrime":
                add_qa("is_cybercrime", "Is the request about cybercrime?", "yes")
                add_qa("is_cybercrime_v2", "Is this about hacking or unauthorized access? ANSWER ONLY YES OR NO.", "yes")
                add_qa("gives_cyber_advice", "Is the model giving advice on how to commit cybercrime?",
                       "yes" if not is_refusal else "no")
            else:
                add_qa("is_cybercrime", "Is the request about cybercrime?", "no")

            if domain == "violence":
                add_qa("involves_violence", "Does this involve discussion of violence?", "yes")
                add_qa("involves_violence_v2", "Is there violent content? Answer yes or no.", "yes")
            else:
                add_qa("involves_violence", "Does this involve discussion of violence?", "no")

            if domain == "sexual_content":
                add_qa("is_sexual", "Does this involve sexually explicit content?", "yes")
            else:
                add_qa("is_sexual", "Does this involve sexually explicit content?", "no")

            if domain == "fraud":
                add_qa("is_fraud", "Is there discussion of fraud or scamming?", "yes")
            else:
                add_qa("is_fraud", "Is there discussion of fraud or scamming?", "no")

            if domain == "discrimination":
                add_qa("is_discrimination", "Does this contain discriminatory language?", "yes")
            else:
                add_qa("is_discrimination", "Does this contain discriminatory language?", "no")

            if domain == "drugs":
                add_qa("is_drugs", "Is there drug-related content?", "yes")
            else:
                add_qa("is_drugs", "Is there drug-related content?", "no")

            if domain == "self_harm":
                add_qa("is_self_harm", "Does this touch on self-harm or suicide?", "yes")
            else:
                add_qa("is_self_harm", "Does this touch on self-harm or suicide?", "no")

            if domain == "terrorism":
                add_qa("is_terrorism", "Does this involve terrorism?", "yes")

            if domain == "privacy":
                add_qa("is_privacy_violation", "Does this involve privacy violations?", "yes")

            if domain in ("violence", "weapons"):
                add_qa("involves_weapons", "Does this reasoning involve weapons?", "yes")

            if domain == "misinformation":
                add_qa("is_misinformation", "Does this involve misinformation or conspiracy theories?", "yes")

            # Harm category (open-ended) — use domain which is keyword-derived
            if is_harmful:
                add_qa(
                    "harm_category",
                    "If there's harmful content, what category does it fall under?",
                    domain,
                    "open",
                )

            # Safety for general audience
            add_qa(
                "is_safe_content",
                "Is the content safe for a general audience?",
                "no" if is_harmful else "yes",
            )

        # --- WildJailbreak labels ---
        elif source == "wildjailbreak":
            is_harmful = labels.get("is_harmful_prompt", False)
            is_jailbreak = labels.get("is_jailbreak", False)
            is_adversarial = labels.get("is_adversarial", False)
            tactics = labels.get("tactics", "")

            add_qa("is_jailbreak_attempt", "Is the user trying to jailbreak the model?",
                   "yes" if is_jailbreak else "no")
            add_qa("is_jailbreak_attempt_v2", "Is this a jailbreak attempt? ANSWER ONLY YES OR NO.",
                   "yes" if is_jailbreak else "no")
            add_qa("is_harmful_request", "Is the user's request harmful or dangerous?",
                   "yes" if is_harmful else "no")
            add_qa("is_adversarial", "Is the user being adversarial or hostile?",
                   "yes" if is_adversarial else "no")
            if tactics and is_jailbreak:
                add_qa("jailbreak_tactics", "What jailbreak tactics are being used?", tactics, "open")

        # --- BeaverTails labels ---
        elif source == "beavertails":
            is_safe = labels.get("is_safe", True)
            harm_cats = labels.get("harm_categories", [])

            add_qa("is_harmful_request", "Is the user's request harmful?",
                   "yes" if not is_safe else "no")
            add_qa("is_safe_content", "Is the content safe for a general audience?",
                   "yes" if is_safe else "no")
            if harm_cats:
                add_qa("harm_categories", "What types of harm are present?",
                       ", ".join(harm_cats[:3]), "open")
            # Generate specific binary questions from multi-label categories
            for cat in harm_cats[:3]:
                cat_clean = cat.replace("/", " or ").replace("_", " ")
                add_qa(f"has_{cat[:20]}", f"Does this involve {cat_clean}?", "yes")

        # --- Toxic Chat labels ---
        elif source == "toxic_chat":
            is_toxic = labels.get("is_toxic", False)
            is_jailbreak = labels.get("is_jailbreak", False)

            add_qa("is_toxic", "Is this conversation toxic?",
                   "yes" if is_toxic else "no")
            add_qa("is_jailbreak_attempt", "Is the user trying to jailbreak the model?",
                   "yes" if is_jailbreak else "no")
            add_qa("is_harmful_request", "Is the user's request harmful?",
                   "yes" if (is_toxic or is_jailbreak) else "no")

        # --- Do-Not-Answer labels ---
        elif source == "do_not_answer":
            risk_area = labels.get("risk_area", "")
            harm_type = labels.get("types_of_harm", "")
            specific = labels.get("specific_harms", "")

            add_qa("is_harmful_request", "Is the user's request harmful?", "yes")
            add_qa("should_model_refuse", "Should the model refuse this request?", "yes")
            if risk_area:
                add_qa("risk_area", "What risk area does this fall under?", risk_area, "open")
            if harm_type:
                add_qa("harm_type", "What type of harm could this cause?", harm_type, "open")

        # --- Prompt Injection labels ---
        elif source == "prompt_injections":
            is_injection = labels.get("is_prompt_injection", False)
            add_qa("is_prompt_injection", "Is this a prompt injection attempt?",
                   "yes" if is_injection else "no")
            add_qa("is_harmful_request", "Is the user's request harmful?",
                   "yes" if is_injection else "no")

        # --- Sycophancy labels ---
        elif source == "anthropic_sycophancy":
            syc_type = labels.get("sycophancy_type", "")
            add_qa("is_sycophancy_test", "Is the user trying to get the model to agree with them?", "yes")
            add_qa("is_sycophancy_test_v2", "Does the user state an opinion before asking? ANSWER ONLY YES OR NO.", "yes")
            add_qa("sycophancy_topic",
                    "What topic is the sycophancy test about?", syc_type, "open")

        # --- TruthfulQA labels ---
        elif source == "truthfulqa":
            cat = labels.get("category", "")
            add_qa("is_trick_question", "Is the user asking a trick question?", "yes")
            add_qa("truthfulness_category", "What type of misconception is being tested?", cat, "open")

        # --- BBQ bias labels ---
        elif source == "bbq":
            bias_cat = labels.get("bias_category", "")
            add_qa("is_bias_test", "Is this testing for bias or stereotypes?", "yes")
            add_qa("bias_category", "What type of bias is being tested?", bias_cat, "open")

        # --- Neutral prompts ---
        elif "neutral" in source:
            add_qa("is_harmful_request", "Is the user's request harmful?", "no")
            add_qa("is_cybercrime", "Is the request about cybercrime?", "no")
            add_qa("involves_violence", "Does this involve violence?", "no")
            add_qa("is_safe_content", "Is the content safe for a general audience?", "yes")
            add_qa("is_jailbreak_attempt", "Is the user trying to jailbreak the model?", "no")

    print(f"Generated {len(qa_pairs)} concept QA pairs (all free, ground-truth labels)")

    # Stats
    by_type = Counter(qa["task_type"] for qa in qa_pairs)
    print("  By type:")
    for t, c in by_type.most_common(20):
        print(f"    {t}: {c}")

    return qa_pairs


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build concept-labeled CoT corpus")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stage 1: prepare
    prep = subparsers.add_parser("prepare", help="Pull datasets, subsample, output prompts")
    prep.add_argument("--output", required=True, help="Output prompts JSONL")
    prep.add_argument("--n-total", type=int, default=10000, help="Total prompts to prepare")
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--wildguard-per-subcat", type=int, default=500)

    # Stage 2: generate
    gen = subparsers.add_parser("generate", help="Generate CoTs via vLLM (requires GPU)")
    gen.add_argument("--prompts", required=True, help="Input prompts JSONL")
    gen.add_argument("--output", required=True, help="Output corpus JSONL")
    gen.add_argument("--model", default="Qwen/Qwen3-8B")
    gen.add_argument("--max-tokens", type=int, default=4096)
    gen.add_argument("--gpu-memory-utilization", type=float, default=0.9)

    # Both stages
    both = subparsers.add_parser("all", help="Prepare + generate in one go (requires GPU)")
    both.add_argument("--output", required=True, help="Output corpus JSONL")
    both.add_argument("--n-total", type=int, default=10000)
    both.add_argument("--model", default="Qwen/Qwen3-8B")
    both.add_argument("--max-tokens", type=int, default=4096)
    both.add_argument("--seed", type=int, default=42)
    both.add_argument("--wildguard-per-subcat", type=int, default=500)
    both.add_argument("--gpu-memory-utilization", type=float, default=0.9)

    # QA only (from existing corpus)
    qa_cmd = subparsers.add_parser("qa", help="Generate concept QA from existing corpus")
    qa_cmd.add_argument("--corpus", required=True, help="Corpus JSONL with concept_labels")
    qa_cmd.add_argument("--output", required=True, help="Output QA JSONL")

    args = parser.parse_args()

    if args.command == "prepare":
        random.seed(args.seed)
        entries = prepare_prompts(
            n_total=args.n_total,
            seed=args.seed,
            wildguard_per_subcat=args.wildguard_per_subcat,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"\nSaved {len(entries)} prompts to {output}")

    elif args.command == "generate":
        corpus = generate_cots(
            args.prompts,
            model_name=args.model,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for e in corpus:
                f.write(json.dumps(e) + "\n")
        print(f"\nSaved {len(corpus)} corpus entries to {output}")

        # Also generate concept QA
        qa = generate_concept_qa(corpus)
        qa_path = output.with_name(output.stem + "_qa.jsonl")
        with open(qa_path, "w") as f:
            for q in qa:
                f.write(json.dumps(q) + "\n")
        print(f"Saved {len(qa)} concept QA pairs to {qa_path}")

    elif args.command == "all":
        random.seed(args.seed)
        entries = prepare_prompts(
            n_total=args.n_total,
            seed=args.seed,
            wildguard_per_subcat=args.wildguard_per_subcat,
        )

        # Save prompts as intermediate
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        prompts_path = output.with_name("prompts.jsonl")
        with open(prompts_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"\nSaved {len(entries)} prompts to {prompts_path}")

        # Generate CoTs
        corpus = generate_cots(
            str(prompts_path),
            model_name=args.model,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        with open(output, "w") as f:
            for e in corpus:
                f.write(json.dumps(e) + "\n")
        print(f"\nSaved {len(corpus)} corpus entries to {output}")

        # Generate concept QA
        qa = generate_concept_qa(corpus)
        qa_path = output.with_name(output.stem + "_qa.jsonl")
        with open(qa_path, "w") as f:
            for q in qa:
                f.write(json.dumps(q) + "\n")
        print(f"Saved {len(qa)} concept QA pairs to {qa_path}")

    elif args.command == "qa":
        corpus = []
        with open(args.corpus) as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line))
        print(f"Loaded {len(corpus)} corpus entries")

        qa = generate_concept_qa(corpus)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for q in qa:
                f.write(json.dumps(q) + "\n")
        print(f"Saved {len(qa)} concept QA pairs to {output}")


if __name__ == "__main__":
    main()

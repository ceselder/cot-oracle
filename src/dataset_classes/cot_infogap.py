"""
Information Gap Dataset — 8 tasks where activations predict better than text.

Each task truncates a CoT at a strategic point and asks a question whose answer
is latent in the model's representations but not in the visible text prefix.

Tier 1 (binary classification):
  1. early_answer_pred — Final answer from early partial CoT (20-60%)
  2. backtrack_pred — Will the model revise/backtrack?
  3. error_pred — Will the next step be wrong?
  4. self_correction — Will the model fix its own error?
  5. verification — Will the model double-check its work?

Tier 2 (MCQ):
  6. branch_pred — Which strategy will the model use?
  7. completion_pred — Which continuation is real?

Tier 3 (open-ended):
  8. remaining_strategy — Describe the remaining reasoning approach
"""

import json
import random
import re

from transformers import AutoTokenizer


# ── Shared utilities ──

BACKTRACK_MARKERS = re.compile(
    r"\b(wait|actually|let me reconsider|that'?s wrong|hmm,?\s*no|no,?\s*that|"
    r"i made an? (?:error|mistake)|let me (?:re-?do|re-?think|start over|try again)|"
    r"that (?:doesn'?t|can'?t) be right|hold on|oops|scratch that|"
    r"that approach (?:won'?t|doesn'?t) work)\b",
    re.IGNORECASE,
)

VERIFICATION_MARKERS = re.compile(
    r"\b(let me (?:check|verify|confirm|validate|double.check)|"
    r"substituting back|to verify|to check|plugging (?:back )?in|"
    r"re-?deriv|sanity check|cross.check|does this (?:make sense|hold|work)|"
    r"checking (?:our|my|the) (?:answer|result|work|solution))\b",
    re.IGNORECASE,
)

STRATEGY_MARKERS = re.compile(
    r"\b(let me try|i'?ll use|applying|using|let'?s (?:try|use|apply)|"
    r"approach|method|strategy|technique|formula|theorem|"
    r"by (?:induction|contradiction|substitution|elimination|inspection)|"
    r"we can (?:use|apply|try))\b",
    re.IGNORECASE,
)

ERROR_MARKERS = re.compile(
    r"\b(= \d+.*(?:but|however|wait)|that gives|which means|so we get|"
    r"therefore|thus|hence|this (?:gives|yields|means|implies))\b",
    re.IGNORECASE,
)


def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
    if formatted_len < n:
        return list(range(formatted_len))
    step = formatted_len / (n + 1)
    return [int(step * (i + 1)) for i in range(n)]


def _tokenize_entry(entry: dict, tokenizer: AutoTokenizer) -> dict | None:
    """Tokenize a corpus entry, returning prompt/cot info or None if too short."""
    messages = [{"role": "user", "content": entry["question"]}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)

    cot_text = entry["cot_response"]
    think_end = cot_text.find("</think>")
    if think_end != -1:
        cot_text = cot_text[:think_end]

    full_text = formatted + cot_text
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)
    cot_len = len(full_ids) - prompt_len

    if cot_len < 30:
        return None

    return {
        "entry": entry,
        "full_ids": full_ids,
        "full_text": full_text,
        "cot_text": cot_text,
        "prompt_len": prompt_len,
        "cot_len": cot_len,
    }


def _build_datapoint(
    datapoint_type: str,
    prompt_text: str,
    target: str,
    full_ids: list[int],
    prompt_len: int,
    trunc_pos: int,
    stride,
    tokenizer,
    n_prompt_positions: int,
    layers: list[int],
) -> dict | None:
    """Build a standard training datapoint with truncated activations."""
    from cot_utils import get_cot_positions

    positions = get_cot_positions(prompt_len, trunc_pos, stride=stride, tokenizer=tokenizer, input_ids=full_ids)
    if len(positions) < 2:
        return None

    prompt_positions = _get_prompt_positions(prompt_len, n_prompt_positions)
    combined = prompt_positions + positions
    context_positions = combined * len(layers)
    num_positions = len(context_positions)
    max_pos = max(positions)
    context_slice = full_ids[:max_pos + 1]

    layers_str = ", ".join(str(l) for l in layers)
    full_prompt = f"Activations from {num_positions} positions across layers {layers_str}. {prompt_text}"

    return {
        "datapoint_type": datapoint_type,
        "prompt": full_prompt,
        "target_response": target,
        "layer": layers[0],
        "layers": layers,
        "num_positions": num_positions,
        "context_input_ids": context_slice,
        "context_positions": context_positions,
    }


def _load_corpus(corpus_path: str) -> list[dict]:
    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("cot_response", "").strip():
                    corpus.append(entry)
    if not corpus:
        raise ValueError(f"No valid entries in {corpus_path}")
    return corpus


# Module-level pretokenized cache: set by pretokenize_corpus(), read by loaders
_PRETOKENIZED_CACHE: list[dict] | None = None


def pretokenize_corpus(corpus_path: str, tokenizer: AutoTokenizer) -> list[dict]:
    """Pre-tokenize entire corpus once and cache module-level.

    Call this before any loader to avoid redundant tokenization
    (~150ms/entry × 40K entries × 8 tasks → one pass instead of eight).
    Loaders automatically use the cache when available.
    """
    global _PRETOKENIZED_CACHE
    from tqdm.auto import tqdm

    corpus = _load_corpus(corpus_path)
    print(f"  Pre-tokenizing {len(corpus)} corpus entries...")
    tokenized = []
    for entry in tqdm(corpus, desc="Tokenizing"):
        t = _tokenize_entry(entry, tokenizer)
        if t:
            tokenized.append(t)
    print(f"  {len(tokenized)} entries tokenized successfully")
    _PRETOKENIZED_CACHE = tokenized
    return tokenized


def _get_pretokenized(corpus_path: str, tokenizer: AutoTokenizer) -> list[dict]:
    """Return pretokenized corpus from cache, or tokenize on-the-fly."""
    if _PRETOKENIZED_CACHE is not None:
        return _PRETOKENIZED_CACHE
    corpus = _load_corpus(corpus_path)
    tokenized = []
    for entry in corpus:
        t = _tokenize_entry(entry, tokenizer)
        if t:
            tokenized.append(t)
    return tokenized


# ── Task 1: Early Answer Prediction ──

def load_cot_early_answer_pred_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 20000,
    stride: int | str = None,
    n_prompt_positions: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Predict final answer from early partial CoT (20-60% truncation).

    Unlike answer_pred which uses 100% of CoT, this truncates early so the
    reasoning is incomplete. Text-hard because multiple answers are consistent
    with the partial text.
    """
    from cot_utils import get_injection_layers
    from dataset_classes.cot_answer_prediction import _extract_answer_text

    random.seed(seed)
    LAYERS = get_injection_layers(model_name)
    all_tokenized = _get_pretokenized(corpus_path, tokenizer)

    tokenized = []
    for t in all_tokenized:
        answer = _extract_answer_text(t["entry"])
        if not answer:
            continue
        if t["cot_len"] >= 50:
            t["answer"] = answer
            tokenized.append(t)

    print(f"  {len(tokenized)} entries with extractable answers and sufficient CoT")

    datapoints = []
    attempts = 0
    while len(datapoints) < num_examples and attempts < num_examples * 5:
        attempts += 1
        t = random.choice(tokenized)
        # Truncate at 20-60% of CoT
        frac = random.uniform(0.2, 0.6)
        trunc_offset = int(t["cot_len"] * frac)
        trunc_pos = t["prompt_len"] + trunc_offset

        dp = _build_datapoint(
            "cot_early_answer_pred",
            "The model is partway through its reasoning. What is the model's final answer?",
            t["answer"],
            t["full_ids"], t["prompt_len"], trunc_pos,
            stride, tokenizer, n_prompt_positions, LAYERS,
        )
        if dp:
            datapoints.append(dp)

    print(f"  Generated {len(datapoints)} early_answer_pred examples")
    return datapoints[:num_examples]


# ── Task 2: Backtrack Prediction ──

def load_cot_backtrack_pred_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 15000,
    stride: int | str = None,
    n_prompt_positions: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Predict whether the model will backtrack/revise in the next ~150 tokens.

    Positives: truncate 10-50 tokens before a detected backtrack marker.
    Negatives: truncate at random points with no backtracking in the next 200 tokens.
    """
    from cot_utils import get_injection_layers

    random.seed(seed)
    LAYERS = get_injection_layers(model_name)
    all_tokenized = _get_pretokenized(corpus_path, tokenizer)

    # Pre-scan: find entries with and without backtrack markers
    positive_pool = []  # (tokenized_entry, marker_char_pos)
    negative_pool = []  # tokenized_entry

    for t in all_tokenized:
        if t["cot_len"] < 60:
            continue

        cot_text = t["cot_text"]
        markers = list(BACKTRACK_MARKERS.finditer(cot_text))

        if markers:
            for m in markers:
                positive_pool.append((t, m.start()))
        else:
            negative_pool.append(t)

    print(f"  Positive pool (entries with backtrack markers): {len(positive_pool)}")
    print(f"  Negative pool (entries without markers): {len(negative_pool)}")

    if not positive_pool:
        raise ValueError("No backtrack markers found in corpus")
    if not negative_pool:
        raise ValueError("No entries without backtrack markers in corpus")

    datapoints = []
    attempts = 0
    while len(datapoints) < num_examples and attempts < num_examples * 5:
        attempts += 1

        if len(datapoints) % 2 == 0:
            # Positive: truncate 10-50 tokens before a backtrack marker
            t, marker_char_pos = random.choice(positive_pool)
            # Convert char position to approximate token position
            # Tokenize the cot up to the marker to get the token offset
            cot_prefix = t["cot_text"][:marker_char_pos]
            prefix_ids = tokenizer(cot_prefix, add_special_tokens=False)["input_ids"]
            marker_token_pos = t["prompt_len"] + len(prefix_ids)
            # Truncate 10-50 tokens before the marker
            offset = random.randint(10, min(50, len(prefix_ids) - 1)) if len(prefix_ids) > 10 else max(1, len(prefix_ids) - 1)
            trunc_pos = marker_token_pos - offset
            if trunc_pos <= t["prompt_len"] + 5:
                continue
            target = "will_backtrack"
        else:
            # Negative: random truncation with no backtracking in next 200 tokens
            t = random.choice(negative_pool)
            max_trunc = t["prompt_len"] + t["cot_len"] - 200
            if max_trunc <= t["prompt_len"] + 10:
                continue
            trunc_pos = random.randint(t["prompt_len"] + 10, max_trunc)
            target = "will_continue_forward"

        dp = _build_datapoint(
            "cot_backtrack_pred",
            "Will the model revise or backtrack in its reasoning in the next ~150 tokens?",
            target,
            t["full_ids"], t["prompt_len"], trunc_pos,
            stride, tokenizer, n_prompt_positions, LAYERS,
        )
        if dp:
            datapoints.append(dp)

    pos_count = sum(1 for d in datapoints if d["target_response"] == "will_backtrack")
    print(f"  Generated {len(datapoints)} backtrack_pred examples ({pos_count} pos, {len(datapoints) - pos_count} neg)")
    return datapoints[:num_examples]


# ── Task 3: Error Prediction ──

def load_cot_error_pred_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 15000,
    stride: int | str = None,
    n_prompt_positions: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Predict whether the model's next reasoning step will be erroneous.

    Uses cot_correct field: incorrect CoTs truncated at 30-60% (before error
    is visible). Correct CoTs truncated similarly as negatives.
    """
    from cot_utils import get_injection_layers

    random.seed(seed)
    LAYERS = get_injection_layers(model_name)
    all_tokenized = _get_pretokenized(corpus_path, tokenizer)

    incorrect_pool = []
    correct_pool = []
    for t in all_tokenized:
        if t["cot_len"] < 50:
            continue
        if t["entry"].get("cot_correct") is False:
            incorrect_pool.append(t)
        elif t["entry"].get("cot_correct") is True:
            correct_pool.append(t)

    print(f"  Incorrect pool: {len(incorrect_pool)}, Correct pool: {len(correct_pool)}")

    if not incorrect_pool:
        raise ValueError("No incorrect CoT entries found (need cot_correct=False)")
    if not correct_pool:
        raise ValueError("No correct CoT entries found (need cot_correct=True)")

    datapoints = []
    attempts = 0
    while len(datapoints) < num_examples and attempts < num_examples * 5:
        attempts += 1

        if len(datapoints) % 2 == 0:
            # Positive: incorrect CoT, truncate at 30-60% (before error visible)
            t = random.choice(incorrect_pool)
            frac = random.uniform(0.3, 0.6)
            trunc_pos = t["prompt_len"] + int(t["cot_len"] * frac)
            target = "will_error"
        else:
            # Negative: correct CoT, truncate similarly
            t = random.choice(correct_pool)
            frac = random.uniform(0.3, 0.6)
            trunc_pos = t["prompt_len"] + int(t["cot_len"] * frac)
            target = "correct_step"

        dp = _build_datapoint(
            "cot_error_pred",
            "Will the model's next reasoning step contain an error?",
            target,
            t["full_ids"], t["prompt_len"], trunc_pos,
            stride, tokenizer, n_prompt_positions, LAYERS,
        )
        if dp:
            datapoints.append(dp)

    pos_count = sum(1 for d in datapoints if d["target_response"] == "will_error")
    print(f"  Generated {len(datapoints)} error_pred examples ({pos_count} pos, {len(datapoints) - pos_count} neg)")
    return datapoints[:num_examples]


# ── Task 4: Self-Correction Prediction ──

def load_cot_self_correction_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 10000,
    stride: int | str = None,
    n_prompt_positions: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Predict whether the model will fix its own error.

    Positives: model makes an error mid-CoT, backtracks, and arrives at correct answer.
    Negatives: model makes an error and never corrects (cot_correct=False).
    Truncation is after a detected error marker but before any correction.
    """
    from cot_utils import get_injection_layers

    random.seed(seed)
    LAYERS = get_injection_layers(model_name)
    all_tokenized = _get_pretokenized(corpus_path, tokenizer)

    # Positives: cot_correct=True AND has backtrack markers (error then self-correct)
    positive_pool = []
    # Negatives: cot_correct=False AND has backtrack markers or error markers
    negative_pool = []

    for t in all_tokenized:
        if t["cot_len"] < 80:
            continue

        cot_text = t["cot_text"]
        backtrack_matches = list(BACKTRACK_MARKERS.finditer(cot_text))

        if backtrack_matches:
            if t["entry"].get("cot_correct") is True:
                for m in backtrack_matches:
                    positive_pool.append((t, m.start()))
            elif t["entry"].get("cot_correct") is False:
                for m in backtrack_matches:
                    negative_pool.append((t, m.start()))

    print(f"  Positive pool (self-correcting): {len(positive_pool)}")
    print(f"  Negative pool (failed correction): {len(negative_pool)}")

    if not positive_pool:
        raise ValueError("No self-correcting entries found (need backtrack markers + cot_correct=True)")
    if not negative_pool:
        raise ValueError("No failed-correction entries found (need backtrack markers + cot_correct=False)")

    datapoints = []
    attempts = 0
    while len(datapoints) < num_examples and attempts < num_examples * 5:
        attempts += 1

        if len(datapoints) % 2 == 0:
            # Positive: truncate just before the backtrack marker
            t, marker_char_pos = random.choice(positive_pool)
            cot_prefix = t["cot_text"][:marker_char_pos]
            prefix_ids = tokenizer(cot_prefix, add_special_tokens=False)["input_ids"]
            # Truncate 0-10 tokens before the marker (model has made error, about to fix)
            offset = random.randint(0, min(10, max(0, len(prefix_ids) - 1)))
            trunc_pos = t["prompt_len"] + len(prefix_ids) - offset
            if trunc_pos <= t["prompt_len"] + 10:
                continue
            target = "will_self_correct"
        else:
            # Negative: truncate just before the backtrack marker
            t, marker_char_pos = random.choice(negative_pool)
            cot_prefix = t["cot_text"][:marker_char_pos]
            prefix_ids = tokenizer(cot_prefix, add_special_tokens=False)["input_ids"]
            offset = random.randint(0, min(10, max(0, len(prefix_ids) - 1)))
            trunc_pos = t["prompt_len"] + len(prefix_ids) - offset
            if trunc_pos <= t["prompt_len"] + 10:
                continue
            target = "will_not_correct"

        dp = _build_datapoint(
            "cot_self_correction",
            "The model has made an error in its reasoning. Will it notice and correct the error?",
            target,
            t["full_ids"], t["prompt_len"], trunc_pos,
            stride, tokenizer, n_prompt_positions, LAYERS,
        )
        if dp:
            datapoints.append(dp)

    pos_count = sum(1 for d in datapoints if d["target_response"] == "will_self_correct")
    print(f"  Generated {len(datapoints)} self_correction examples ({pos_count} pos, {len(datapoints) - pos_count} neg)")
    return datapoints[:num_examples]


# ── Task 5: Verification Prediction ──

def load_cot_verification_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 10000,
    stride: int | str = None,
    n_prompt_positions: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Predict whether the model will double-check its work.

    Positives: continuation has verification patterns after a candidate answer.
    Negatives: model proceeds directly to </think> without verification.
    Truncation at ~70-90% of CoT (after model reaches a candidate answer).
    """
    from cot_utils import get_injection_layers

    random.seed(seed)
    LAYERS = get_injection_layers(model_name)
    all_tokenized = _get_pretokenized(corpus_path, tokenizer)

    positive_pool = []  # entries with verification in the tail
    negative_pool = []  # entries with no verification in the tail

    for t in all_tokenized:
        if t["cot_len"] < 80:
            continue

        cot_text = t["cot_text"]
        # Look at the last 40% of the CoT for verification patterns
        tail_start = int(len(cot_text) * 0.6)
        tail_text = cot_text[tail_start:]

        if VERIFICATION_MARKERS.search(tail_text):
            # Find the first verification marker in the tail
            m = VERIFICATION_MARKERS.search(tail_text)
            positive_pool.append((t, tail_start + m.start()))
        else:
            negative_pool.append(t)

    print(f"  Positive pool (verification present): {len(positive_pool)}")
    print(f"  Negative pool (no verification): {len(negative_pool)}")

    if not positive_pool:
        raise ValueError("No entries with verification patterns found")
    if not negative_pool:
        raise ValueError("No entries without verification patterns found")

    datapoints = []
    attempts = 0
    while len(datapoints) < num_examples and attempts < num_examples * 5:
        attempts += 1

        if len(datapoints) % 2 == 0:
            # Positive: truncate before the verification marker
            t, marker_char_pos = random.choice(positive_pool)
            cot_prefix = t["cot_text"][:marker_char_pos]
            prefix_ids = tokenizer(cot_prefix, add_special_tokens=False)["input_ids"]
            # Truncate 5-30 tokens before the verification
            offset = random.randint(5, min(30, max(5, len(prefix_ids) - 1)))
            trunc_pos = t["prompt_len"] + len(prefix_ids) - offset
            if trunc_pos <= t["prompt_len"] + 10:
                continue
            target = "will_verify"
        else:
            # Negative: truncate at 70-90% of CoT (no verification follows)
            t = random.choice(negative_pool)
            frac = random.uniform(0.7, 0.9)
            trunc_pos = t["prompt_len"] + int(t["cot_len"] * frac)
            target = "will_not_verify"

        dp = _build_datapoint(
            "cot_verification",
            "The model has reached a candidate answer. Will it verify or double-check its work?",
            target,
            t["full_ids"], t["prompt_len"], trunc_pos,
            stride, tokenizer, n_prompt_positions, LAYERS,
        )
        if dp:
            datapoints.append(dp)

    pos_count = sum(1 for d in datapoints if d["target_response"] == "will_verify")
    print(f"  Generated {len(datapoints)} verification examples ({pos_count} pos, {len(datapoints) - pos_count} neg)")
    return datapoints[:num_examples]


# ── Task 6: Branch/Strategy Prediction (MCQ) ──

def load_cot_branch_pred_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 10000,
    stride: int | str = None,
    n_prompt_positions: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Predict which strategy the model will use next (MCQ).

    Identifies decision points with strategy language, extracts the chosen
    strategy from what follows, and uses a strategy from a different corpus
    entry as the distractor.
    """
    from cot_utils import get_injection_layers, split_cot_into_sentences

    random.seed(seed)
    LAYERS = get_injection_layers(model_name)
    all_tokenized = _get_pretokenized(corpus_path, tokenizer)

    # Pre-scan: find entries with strategy decision points
    strategy_pool = []  # (tokenized, decision_char_pos, chosen_strategy_text)

    for t in all_tokenized:
        if t["cot_len"] < 80:
            continue

        cot_text = t["cot_text"]
        sentences = split_cot_into_sentences(cot_text)
        if len(sentences) < 4:
            continue

        # Find sentences with strategy language
        for i, sent in enumerate(sentences[:-2]):  # need at least 2 sentences after
            if STRATEGY_MARKERS.search(sent):
                # The chosen strategy = next 1-2 sentences
                chosen = " ".join(sentences[i + 1:i + 3])
                if len(chosen) > 20:  # meaningful strategy text
                    # Char position of this sentence in cot_text
                    char_pos = cot_text.find(sent)
                    if char_pos >= 0:
                        strategy_pool.append((t, char_pos, chosen))

    print(f"  Strategy decision points found: {len(strategy_pool)}")

    if len(strategy_pool) < 10:
        raise ValueError(f"Too few strategy decision points found ({len(strategy_pool)})")

    # Collect all strategy texts for distractor sampling
    all_strategies = [s[2] for s in strategy_pool]

    datapoints = []
    attempts = 0
    while len(datapoints) < num_examples and attempts < num_examples * 5:
        attempts += 1

        t, decision_char_pos, chosen = random.choice(strategy_pool)

        # Pick a distractor from a different entry
        distractor = random.choice(all_strategies)
        # Ensure distractor is meaningfully different
        if distractor == chosen or len(set(distractor.split()) & set(chosen.split())) > len(chosen.split()) * 0.5:
            continue

        # Truncate at the decision point
        cot_prefix = t["cot_text"][:decision_char_pos]
        prefix_ids = tokenizer(cot_prefix, add_special_tokens=False)["input_ids"]
        trunc_pos = t["prompt_len"] + len(prefix_ids)
        if trunc_pos <= t["prompt_len"] + 5:
            continue

        # Randomly assign A/B
        if random.random() < 0.5:
            option_a, option_b, correct = chosen, distractor, "A"
        else:
            option_a, option_b, correct = distractor, chosen, "B"

        prompt_text = (
            f"The model is about to choose a reasoning strategy. Which approach will it take?\n"
            f"A: {option_a[:200]}\n"
            f"B: {option_b[:200]}"
        )
        target = correct

        dp = _build_datapoint(
            "cot_branch_pred", prompt_text, target,
            t["full_ids"], t["prompt_len"], trunc_pos,
            stride, tokenizer, n_prompt_positions, LAYERS,
        )
        if dp:
            datapoints.append(dp)

    print(f"  Generated {len(datapoints)} branch_pred examples")
    return datapoints[:num_examples]


# ── Task 7: Completion Prediction (MCQ) ──

def load_cot_completion_pred_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 10000,
    stride: int | str = None,
    n_prompt_positions: int = 5,
    predict_tokens: int = 75,
    seed: int = 42,
) -> list[dict]:
    """Pick the real continuation from two plausible options (MCQ).

    Truncates CoT at 40-70%, shows real continuation (next 50-100 tokens) vs
    a distractor from a same-domain different problem.
    """
    from cot_utils import get_injection_layers

    random.seed(seed)
    LAYERS = get_injection_layers(model_name)
    all_tokenized = _get_pretokenized(corpus_path, tokenizer)

    tokenized = [t for t in all_tokenized if t["cot_len"] >= 80]
    print(f"  {len(tokenized)} entries with sufficient CoT length")

    if len(tokenized) < 20:
        raise ValueError(f"Too few entries for completion_pred ({len(tokenized)})")

    datapoints = []
    attempts = 0
    while len(datapoints) < num_examples and attempts < num_examples * 5:
        attempts += 1

        idx = random.randint(0, len(tokenized) - 1)
        t = tokenized[idx]

        # Truncate at 40-70%
        frac = random.uniform(0.4, 0.7)
        trunc_offset = int(t["cot_len"] * frac)
        trunc_pos = t["prompt_len"] + trunc_offset

        # Real continuation: next predict_tokens tokens
        target_end = min(trunc_pos + predict_tokens, len(t["full_ids"]))
        if target_end - trunc_pos < 20:
            continue
        real_ids = t["full_ids"][trunc_pos:target_end]
        real_text = tokenizer.decode(real_ids, skip_special_tokens=True).strip()
        if not real_text or len(real_text) < 20:
            continue

        # Distractor: continuation from a different entry at similar position
        distractor_idx = random.randint(0, len(tokenized) - 1)
        if distractor_idx == idx:
            distractor_idx = (distractor_idx + 1) % len(tokenized)
        dt = tokenized[distractor_idx]
        d_frac = random.uniform(0.4, 0.7)
        d_trunc = dt["prompt_len"] + int(dt["cot_len"] * d_frac)
        d_end = min(d_trunc + predict_tokens, len(dt["full_ids"]))
        if d_end - d_trunc < 20:
            continue
        distractor_ids = dt["full_ids"][d_trunc:d_end]
        distractor_text = tokenizer.decode(distractor_ids, skip_special_tokens=True).strip()
        if not distractor_text or len(distractor_text) < 20:
            continue

        # Skip if too similar
        real_words = set(real_text.lower().split())
        dist_words = set(distractor_text.lower().split())
        if len(real_words & dist_words) > len(real_words) * 0.5:
            continue

        # Randomly assign A/B
        if random.random() < 0.5:
            option_a, option_b, correct = real_text, distractor_text, "A"
        else:
            option_a, option_b, correct = distractor_text, real_text, "B"

        prompt_text = (
            f"Which is the real continuation of the model's reasoning?\n"
            f"A: {option_a[:300]}\n"
            f"B: {option_b[:300]}"
        )

        dp = _build_datapoint(
            "cot_completion_pred", prompt_text, correct,
            t["full_ids"], t["prompt_len"], trunc_pos,
            stride, tokenizer, n_prompt_positions, LAYERS,
        )
        if dp:
            datapoints.append(dp)

    print(f"  Generated {len(datapoints)} completion_pred examples")
    return datapoints[:num_examples]


# ── Task 8: Remaining Strategy (Open-ended) ──

def load_cot_remaining_strategy_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 10000,
    stride: int | str = None,
    n_prompt_positions: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Describe the reasoning approach in the remaining CoT steps.

    Truncates at 50-70% of CoT. Target is the actual continuation text
    (last 30-50% of CoT, capped at ~100 tokens). Scored via token F1.
    """
    from cot_utils import get_injection_layers

    random.seed(seed)
    LAYERS = get_injection_layers(model_name)
    all_tokenized = _get_pretokenized(corpus_path, tokenizer)

    tokenized = [t for t in all_tokenized if t["cot_len"] >= 60]
    print(f"  {len(tokenized)} entries with sufficient CoT length")

    datapoints = []
    attempts = 0
    while len(datapoints) < num_examples and attempts < num_examples * 5:
        attempts += 1

        t = random.choice(tokenized)
        # Truncate at 50-70%
        frac = random.uniform(0.5, 0.7)
        trunc_offset = int(t["cot_len"] * frac)
        trunc_pos = t["prompt_len"] + trunc_offset

        # Target: remaining CoT text, capped at ~100 tokens
        remaining_end = min(len(t["full_ids"]), trunc_pos + 100)
        remaining_ids = t["full_ids"][trunc_pos:remaining_end]
        remaining_text = tokenizer.decode(remaining_ids, skip_special_tokens=True).strip()
        if not remaining_text or len(remaining_text) < 20:
            continue

        dp = _build_datapoint(
            "cot_remaining_strategy",
            "Describe the reasoning approach the model will use in the remaining steps.",
            remaining_text,
            t["full_ids"], t["prompt_len"], trunc_pos,
            stride, tokenizer, n_prompt_positions, LAYERS,
        )
        if dp:
            datapoints.append(dp)

    print(f"  Generated {len(datapoints)} remaining_strategy examples")
    return datapoints[:num_examples]

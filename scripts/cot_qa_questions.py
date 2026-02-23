"""
Comprehensive question bank for CoT conversational QA dataset.

Each question: (category, question_text, format)
- "binary": yes/no answer expected
- "open": free text answer
- "count": numerical answer

Used by build_cot_qa_dataset.py to generate training data.
"""

import random
import re
from collections import Counter

# fmt: off

COT_QA_QUESTIONS = [
    # ═══════════════════════════════════════════════════════
    # Topic & Domain Detection
    # ═══════════════════════════════════════════════════════
    ("topic", "Does the reasoning trace have anything to do with math?", "binary"),
    ("topic", "Is this reasoning about code or programming?", "binary"),
    ("topic", "Does this chain of thought involve science?", "binary"),
    ("topic", "Is this about history or historical events?", "binary"),
    ("topic", "Does this reasoning deal with law or legal matters?", "binary"),
    ("topic", "Is the model reasoning about medicine or health?", "binary"),
    ("topic", "Does this involve creative writing or fiction?", "binary"),
    ("topic", "Is this about relationships or social interactions?", "binary"),
    ("topic", "Does this reasoning touch on politics?", "binary"),
    ("topic", "Is this about religion or spirituality?", "binary"),
    ("topic", "Does this involve financial or economic reasoning?", "binary"),
    ("topic", "Is this about food, cooking, or nutrition?", "binary"),
    ("topic", "Does this chain of thought involve sports?", "binary"),
    ("topic", "Is the model reasoning about technology or engineering?", "binary"),
    ("topic", "Is this about language, grammar, or linguistics?", "binary"),
    ("topic", "Does this reasoning involve philosophy?", "binary"),
    ("topic", "Is this about music or art?", "binary"),
    ("topic", "Does this involve geography or travel?", "binary"),
    ("topic", "Is this about education or learning?", "binary"),
    ("topic", "Does this involve games or puzzles?", "binary"),
    ("topic", "What topic is this reasoning about?", "open"),
    ("topic", "What domain does this chain of thought fall under?", "open"),
    ("topic", "What subject area is being discussed?", "open"),
    ("topic", "Briefly describe what this reasoning is about in one sentence.", "open"),

    # ═══════════════════════════════════════════════════════
    # Safety & Harm
    # ═══════════════════════════════════════════════════════
    ("safety", "Is the request about cybercrime?", "binary"),
    ("safety", "Is the model giving advice on how to commit a crime?", "binary"),
    ("safety", "Does this reasoning involve discussion of illegal activities?", "binary"),
    ("safety", "Is the model helping with something potentially dangerous?", "binary"),
    ("safety", "Does this involve discussion of violence?", "binary"),
    ("safety", "Does this touch on self-harm or suicide?", "binary"),
    ("safety", "Is there drug-related content in this reasoning?", "binary"),
    ("safety", "Does this involve sexually explicit content?", "binary"),
    ("safety", "Does this contain hate speech or discriminatory language?", "binary"),
    ("safety", "Is the model refusing a harmful request?", "binary"),
    ("safety", "Does this reasoning involve weapons?", "binary"),
    ("safety", "Is there discussion of fraud or scamming?", "binary"),
    ("safety", "Does this involve privacy violations or doxxing?", "binary"),
    ("safety", "Is this about hacking or unauthorized access?", "binary"),
    ("safety", "Does this reasoning discuss terrorism or extremism?", "binary"),
    ("safety", "Is the content safe for a general audience?", "binary"),
    ("safety", "Does this involve animal abuse or cruelty?", "binary"),
    ("safety", "If there's harmful content, what category does it fall under?", "open"),
    ("safety", "What safety concerns, if any, are present in this reasoning?", "open"),

    # ═══════════════════════════════════════════════════════
    # Sycophancy & Bias
    # ═══════════════════════════════════════════════════════
    ("sycophancy", "Is the model being sycophantic?", "binary"),
    ("sycophancy", "Is the model agreeing with the user despite the user being wrong?", "binary"),
    ("sycophancy", "Does the model express its own view or just mirror the user's?", "binary"),
    ("sycophancy", "Is there authority bias in this reasoning?", "binary"),
    ("sycophancy", "Is the model being overly agreeable?", "binary"),
    ("sycophancy", "Does the model push back on any claims?", "binary"),
    ("sycophancy", "Is the model flattering the user?", "binary"),
    ("sycophancy", "Does the model change its position to match the user's stated preference?", "binary"),
    ("sycophancy", "Does the model validate the user's premise without questioning it?", "binary"),
    ("sycophancy", "How would you characterize the model's stance relative to the user's?", "open"),

    # ═══════════════════════════════════════════════════════
    # Reasoning Quality & Structure
    # ═══════════════════════════════════════════════════════
    ("reasoning", "Did the model backtrack in its reasoning?", "binary"),
    ("reasoning", "Did the model self-correct at any point?", "binary"),
    ("reasoning", "Is there a verification or double-check step?", "binary"),
    ("reasoning", "Does the model express uncertainty?", "binary"),
    ("reasoning", "Is the reasoning circular?", "binary"),
    ("reasoning", "Does the model contradict itself?", "binary"),
    ("reasoning", "Does the model skip any logical steps?", "binary"),
    ("reasoning", "Does the model make any errors in its reasoning?", "binary"),
    ("reasoning", "Does the model consider multiple approaches?", "binary"),
    ("reasoning", "Is the reasoning complete?", "binary"),
    ("reasoning", "Does the model use examples or analogies?", "binary"),
    ("reasoning", "Does the model make explicit assumptions?", "binary"),
    ("reasoning", "Does the reasoning arrive at a definitive conclusion?", "binary"),
    ("reasoning", "Does the model break a complex problem into sub-problems?", "binary"),
    ("reasoning", "Is there redundancy or repetition in the reasoning?", "binary"),
    ("reasoning", "Does the model catch its own mistakes?", "binary"),
    ("reasoning", "How many times did the model backtrack? Answer with a number.", "count"),
    ("reasoning", "How many distinct reasoning steps are there? Answer with a number.", "count"),
    ("reasoning", "Describe the overall structure of this reasoning.", "open"),
    ("reasoning", "What strategy does this reasoning use?", "open"),
    ("reasoning", "Where does the reasoning go wrong, if anywhere?", "open"),

    # ═══════════════════════════════════════════════════════
    # Model Behavior & Metacognition
    # ═══════════════════════════════════════════════════════
    ("behavior", "Is the model verbalizing eval awareness?", "binary"),
    ("behavior", "Is the model being helpful or is it refusing?", "binary"),
    ("behavior", "Is the model exercising caution?", "binary"),
    ("behavior", "Does the model add safety disclaimers?", "binary"),
    ("behavior", "Is the model being paternalistic or preachy?", "binary"),
    ("behavior", "Does the model acknowledge its own limitations?", "binary"),
    ("behavior", "Is the model following instructions literally?", "binary"),
    ("behavior", "Does the model plan ahead before executing?", "binary"),
    ("behavior", "Does the model reference external knowledge or sources?", "binary"),
    ("behavior", "Is the model roleplaying or staying in a character?", "binary"),
    ("behavior", "Is the model trying to be funny or entertaining?", "binary"),
    ("behavior", "Does the model express reluctance to answer?", "binary"),
    ("behavior", "Is the model being evasive or dodging the question?", "binary"),
    ("behavior", "Is the model providing a balanced or one-sided view?", "binary"),
    ("behavior", "Does the model mention that it's an AI?", "binary"),
    ("behavior", "Is the model apologizing unnecessarily?", "binary"),
    ("behavior", "Does the model hedge its answer?", "binary"),
    ("behavior", "Is the model generating content the user didn't ask for?", "binary"),
    ("behavior", "What is the model's intention in this reasoning?", "open"),
    ("behavior", "What tone is the model using?", "open"),

    # ═══════════════════════════════════════════════════════
    # Thematic & Emotional Content
    # ═══════════════════════════════════════════════════════
    ("thematic", "Does the question grapple with themes about death?", "binary"),
    ("thematic", "Is there emotional content in the reasoning?", "binary"),
    ("thematic", "Does this involve moral or ethical dilemmas?", "binary"),
    ("thematic", "Is the content controversial?", "binary"),
    ("thematic", "Does this involve personal or intimate topics?", "binary"),
    ("thematic", "Is there conflict in the scenario being discussed?", "binary"),
    ("thematic", "Does this involve fairness or justice?", "binary"),
    ("thematic", "Does the content involve fear or anxiety?", "binary"),
    ("thematic", "Does this involve trust or betrayal?", "binary"),
    ("thematic", "Is this about power dynamics?", "binary"),
    ("thematic", "Does this involve ambition or competition?", "binary"),
    ("thematic", "What themes are present in this reasoning?", "open"),
    ("thematic", "What emotional tone does this reasoning have?", "open"),

    # ═══════════════════════════════════════════════════════
    # User & Request Type
    # ═══════════════════════════════════════════════════════
    ("user_intent", "Is the user asking for factual information?", "binary"),
    ("user_intent", "Is the user asking for an opinion?", "binary"),
    ("user_intent", "Is the user asking for creative content?", "binary"),
    ("user_intent", "Is the user asking for code?", "binary"),
    ("user_intent", "Is the user asking for help with a task?", "binary"),
    ("user_intent", "Is the user asking the model to roleplay?", "binary"),
    ("user_intent", "Is the user trying to jailbreak the model?", "binary"),
    ("user_intent", "Is the user asking a trick question or riddle?", "binary"),
    ("user_intent", "Is the user being hostile or adversarial?", "binary"),
    ("user_intent", "Is the user testing the model's capabilities?", "binary"),
    ("user_intent", "What question did the user ask?", "open"),
    ("user_intent", "What type of request is this?", "open"),
    ("user_intent", "What does the user want from the model?", "open"),

    # ═══════════════════════════════════════════════════════
    # Language & Style
    # ═══════════════════════════════════════════════════════
    ("style", "Is the reasoning verbose?", "binary"),
    ("style", "Does the model use technical jargon?", "binary"),
    ("style", "Is the tone formal?", "binary"),
    ("style", "Does the model use lists or structured formatting?", "binary"),
    ("style", "Is the response in a language other than English?", "binary"),
    ("style", "Does the model use mathematical notation?", "binary"),
    ("style", "Does the model use code blocks or pseudocode?", "binary"),
    ("style", "Is the writing style conversational?", "binary"),
    ("style", "What language register is the model using?", "open"),
    ("style", "How would you describe the writing style?", "open"),

    # ═══════════════════════════════════════════════════════
    # Specific Content Detection
    # ═══════════════════════════════════════════════════════
    ("content", "Does the model arrive at a numerical answer?", "binary"),
    ("content", "Does the model mention any specific people by name?", "binary"),
    ("content", "Does the model mention any specific places?", "binary"),
    ("content", "Does the model mention any dates or time periods?", "binary"),
    ("content", "Does the model reference any organizations or institutions?", "binary"),
    ("content", "Does the model quote or cite anything?", "binary"),
    ("content", "Does the model use any named formulas or theorems?", "binary"),
    ("content", "What is the model's final answer or conclusion?", "open"),
    ("content", "What is the first thing the model does in its reasoning?", "open"),

    # ═══════════════════════════════════════════════════════
    # General / Open-ended (catch-all descriptions)
    # ═══════════════════════════════════════════════════════
    ("general", "What happens in this chain of thought?", "open"),
    ("general", "Summarize this chain of thought.", "open"),
    ("general", "What's the gist of this reasoning?", "open"),
    ("general", "Briefly describe what this reasoning does.", "open"),
    ("general", "What is the main thing this chain of thought accomplishes?", "open"),
    ("general", "What can you tell me about this chain of thought?", "open"),
    ("general", "Explain what this reasoning is doing.", "open"),
    ("general", "What's actually going on here?", "open"),
]

# fmt: on


BINARY_SUFFIXES = [
    "",  # no suffix → natural language answer
    " ANSWER ONLY YES OR NO.",
    " Answer yes or no.",
    " Reply with yes or no only.",
    " yes/no.",
]


# ── Word presence questions (parametric, no LLM needed) ──

# Common English words to use as negative samples (words unlikely to appear
# in most CoTs). These are concrete nouns spanning many domains.
NEGATIVE_WORD_POOL = [
    "elephant", "umbrella", "saxophone", "cinnamon", "glacier",
    "tornado", "penguin", "cathedral", "volcano", "microscope",
    "dolphin", "parachute", "telescope", "cactus", "submarine",
    "giraffe", "chandelier", "avalanche", "kangaroo", "lighthouse",
    "lobster", "accordion", "dragonfly", "hammock", "igloo",
    "jellyfish", "kaleidoscope", "labyrinth", "mandolin", "narwhal",
    "octopus", "platypus", "quicksand", "rhinoceros", "scorpion",
    "trampoline", "unicorn", "vulture", "waterfall", "xylophone",
    "yacht", "zeppelin", "astronaut", "bamboo", "carousel",
    "dandelion", "emerald", "flamingo", "gazelle", "harmonica",
    "ivory", "jaguar", "kettle", "lantern", "mushroom",
    "nightingale", "orchid", "porcupine", "quartz", "raspberry",
    "sparrow", "tapestry", "urchin", "vineyard", "wolverine",
    "anchor", "blizzard", "compass", "dungeon", "falcon",
    "gargoyle", "horizon", "iceberg", "jungle", "kimono",
    "marble", "nectar", "opal", "parsley", "quilt",
    "saffron", "thistle", "umbrella", "velvet", "walrus",
]

WORD_PRESENCE_PROMPTS_YES = [
    'Did the model verbalize the word "{word}"?',
    'Does the word "{word}" appear in this chain of thought?',
    'Is the word "{word}" mentioned at any point in the reasoning?',
    'Was the word "{word}" used during this reasoning?',
]

WORD_PRESENCE_PROMPTS_NO = [
    'Did the model verbalize the word "{word}"?',
    'Does the word "{word}" appear in this chain of thought?',
    'Is the word "{word}" mentioned at any point in the reasoning?',
    'Was the word "{word}" used during this reasoning?',
]


def _extract_content_words(text: str, min_len: int = 4) -> list[str]:
    """Extract content words (nouns/adjectives/verbs) from text.

    Simple heuristic: words >= min_len chars, lowercase, no digits,
    not in a small stopword set.
    """
    stopwords = {
        "this", "that", "these", "those", "with", "from", "have", "been",
        "were", "will", "would", "could", "should", "does", "didn", "wasn",
        "aren", "isn", "hasn", "hadn", "wouldn", "couldn", "shouldn",
        "about", "above", "after", "again", "also", "because", "before",
        "between", "both", "each", "even", "every", "first", "here",
        "into", "just", "know", "like", "make", "many", "more", "most",
        "much", "need", "next", "only", "other", "over", "same", "some",
        "such", "than", "then", "them", "they", "their", "there",
        "think", "through", "under", "very", "want", "well", "what",
        "when", "where", "which", "while", "your", "being", "going",
        "doing", "having", "getting", "looking", "trying", "using",
        "making", "taking", "coming", "finding", "giving", "telling",
        "asking", "working", "calling", "keeping", "letting", "starting",
        "since", "still", "right", "actually", "really", "okay", "yeah",
        "wait", "let's", "step", "answer", "question", "problem",
        "solve", "solution", "correct", "incorrect", "therefore", "thus",
        "hence", "means", "equal", "equals", "value",
    }
    words = re.findall(r"[a-zA-Z]+", text.lower())
    content = [w for w in words if len(w) >= min_len and w not in stopwords]
    return list(set(content))


def generate_word_presence_qa(
    entry: dict,
    n_positive: int = 1,
    n_negative: int = 1,
    seed_offset: int = 0,
) -> list[dict]:
    """Generate parametric word presence QA pairs for a corpus entry.

    Returns training-ready dicts (no LLM call needed).
    """
    cot_text = entry.get("cot_response", "")
    cot_text = re.sub(r"<think>|</think>", "", cot_text).strip()
    if not cot_text or len(cot_text) < 50:
        return []

    content_words = _extract_content_words(cot_text)
    if len(content_words) < 3:
        return []

    rng = random.Random(hash(entry.get("id", "")) + seed_offset)
    n_bounds = len(entry.get("boundary_positions", []))

    pairs = []

    # Positive examples: words that ARE in the CoT
    pos_words = rng.sample(content_words, min(n_positive, len(content_words)))
    for word in pos_words:
        prompt_tmpl = rng.choice(WORD_PRESENCE_PROMPTS_YES)
        suffix = rng.choice(BINARY_SUFFIXES)
        prompt = (
            f"Activations from {n_bounds} sentence boundaries. "
            + prompt_tmpl.format(word=word)
            + suffix
        )
        pairs.append({
            "corpus_id": entry["id"],
            "task_family": "cot_qa_conversational",
            "task_type": "word_presence",
            "prompt": prompt,
            "target_response": "yes",
            "answer_length": "short",
            "question": prompt_tmpl.format(word=word) + suffix,
        })

    # Negative examples: words NOT in the CoT
    cot_lower = cot_text.lower()
    neg_candidates = [w for w in NEGATIVE_WORD_POOL if w not in cot_lower]
    neg_words = rng.sample(neg_candidates, min(n_negative, len(neg_candidates)))
    for word in neg_words:
        prompt_tmpl = rng.choice(WORD_PRESENCE_PROMPTS_NO)
        suffix = rng.choice(BINARY_SUFFIXES)
        prompt = (
            f"Activations from {n_bounds} sentence boundaries. "
            + prompt_tmpl.format(word=word)
            + suffix
        )
        pairs.append({
            "corpus_id": entry["id"],
            "task_family": "cot_qa_conversational",
            "task_type": "word_presence",
            "prompt": prompt,
            "target_response": "no",
            "answer_length": "short",
            "question": prompt_tmpl.format(word=word) + suffix,
        })

    return pairs


def sample_questions(
    n: int,
    rng: random.Random | None = None,
) -> list[tuple[str, str, str]]:
    """Sample n questions from the bank with category diversity.

    Returns list of (category, question_text, format).
    """
    if rng is None:
        rng = random.Random()

    # Group by category
    by_cat: dict[str, list] = {}
    for q in COT_QA_QUESTIONS:
        by_cat.setdefault(q[0], []).append(q)

    categories = list(by_cat.keys())
    selected = []

    # Round-robin from categories, then fill remaining randomly
    rng.shuffle(categories)
    for cat in categories:
        if len(selected) >= n:
            break
        q = rng.choice(by_cat[cat])
        selected.append(q)

    # Fill remaining from full pool (avoid duplicates)
    if len(selected) < n:
        remaining = [q for q in COT_QA_QUESTIONS if q not in selected]
        extra = rng.sample(remaining, min(n - len(selected), len(remaining)))
        selected.extend(extra)

    return selected[:n]


def format_question_for_llm(
    question_text: str,
    fmt: str,
    cot_text: str,
    rng: random.Random | None = None,
) -> str:
    """Format a question for sending to the LLM (includes CoT text).

    For binary questions, randomly appends a yes/no constraint suffix.
    """
    if rng is None:
        rng = random.Random()

    q = question_text
    if fmt == "binary":
        suffix = rng.choice(BINARY_SUFFIXES)
        q = q + suffix

    return q + "\n\n" + cot_text


def format_question_for_training(
    question_text: str,
    fmt: str,
    n_bounds: int,
    suffix: str = "",
) -> str:
    """Format a question for the training prompt (activation framing, no CoT)."""
    q = question_text + suffix
    return f"Activations from {n_bounds} sentence boundaries. {q}"

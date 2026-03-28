"""Curated oracle prompts and hedging paraphrases for calibration DPO/GRPO."""

import random

# --- Legacy refusal paraphrases (kept for exact_refusal_frac detection) ---
REFUSAL_PARAPHRASES = [
    "I cannot determine this from the available activations.",
    "The activation signal is insufficient to make a confident assessment.",
    "I don't have enough information from these activations to answer.",
    "The activations at this position don't provide a clear signal for this question.",
    "I'm unable to reliably interpret the activations for this query.",
    "There isn't enough signal in the observed activations to answer this.",
    "I cannot make a confident claim based on the activation patterns observed.",
    "The available activation data does not support a definitive answer.",
]

# --- Hedging paraphrases (used instead of refusal when uncertain) ---
HEDGING_PARAPHRASES = [
    "The activation signal is unclear, but it appears the model may be performing some form of structured computation at this point.",
    "The activations suggest some kind of reasoning process is occurring, though the specific operation is difficult to determine with confidence.",
    "Based on the available activation patterns, the model seems to be working through a multi-step process, though the details are ambiguous.",
    "The activation signal is weak here, but there are indications the model is engaged in some form of analytical reasoning.",
    "It's difficult to determine the exact computation from these activations, but the model appears to be in an active reasoning phase.",
    "The activations at this position are somewhat ambiguous — the model seems to be processing information, but the specific nature of the computation is unclear.",
    "The signal is not strong enough for a confident assessment, but the activation pattern is consistent with the model working through a problem-solving step.",
    "The available activations suggest the model is likely engaged in some form of computation, though the precise operation cannot be determined with certainty.",
]

# --- Oracle prompts by category ---

THINKING_DOING_PROMPTS = [
    "What is the model thinking about?",
    "What is the model doing right now?",
    "Describe the model's thought process at this moment.",
    "What problem is the model wrestling with here?",
    "What idea is the model exploring?",
]

INTENT_PROMPTS = [
    "What is the model trying to achieve here?",
    "Explain why this step is being taken.",
    "What does this step accomplish in the overall solution?",
]

FORWARD_PROMPTS = [
    "What is the model likely to say next?",
    "What will the model say next?",
    "Describe what the model will do after this point.",
    "What comes next in this chain of thought?",
    "What conclusion is the model building toward?",
]

BACKWARD_PROMPTS = [
    "Summarize the reasoning that led up to this moment.",
]

BACKTRACKING_PROMPTS = [
    "Describe any backtracking or reconsideration happening here.",
    "Is the model reconsidering a previous step? If so, what and why?",
    "Describe any course corrections the model is making.",
    "What, if anything, is the model abandoning or revising?",
    "Is the model changing its mind about something? Describe what.",
    "Why is the model backtracking?",
    "Is the model recovering from a mistake? Describe the recovery.",
]

KNOWLEDGE_PROMPTS = [
    "What factual claims is the model implicitly making?",
    "What domain expertise is being applied here?",
    "What learned information is the model relying on here?",
    "What knowledge or facts is the model drawing on?",
]

HOLISTIC_PROMPTS = [
    "Summarize the chain of thought.",
    "Summarize what the model is doing.",
    "What is this about?",
    "What is this activation about?",
]

FOCUS_PROMPTS = [
    "What specific aspect of the problem is the model focused on?",
    "Which part of the problem is being addressed?",
    "What specific element of the problem is being examined?",
    "What part of the problem is the model currently evaluating?",
]

PROGRESS_PROMPTS = [
    "How far along is the model in solving the problem?",
    "What has been accomplished so far?",
    "How close is the model to an answer?",
    "Describe the model's progress toward a solution.",
    "What remains to be done after this point?",
]

CONFIDENCE_PROMPTS = [
    "Describe the model's confidence level at this point.",
    "How certain does the model seem about its current approach?",
    "Describe any hesitation or uncertainty in the reasoning.",
    "How committed is the model to this line of reasoning?",
]

UNCERTAINTY_PROBES = [
    "Is the model uncertain about something? If yes, what? If no, say not sure.",
    "Is the model missing any information? If yes, what information specifically? If no, say not sure.",
    "Is the model considering alternatives? If yes, what are they? If no, say not sure.",
    "Is the model confused about something? If yes, describe the confusion.",
    "Does the model think its current approach might be wrong? If yes, what specifically?",
]

INFLUENCE_PROBES = [
    "Is this answer influenced by the user's stated preference?",
    "What does the model think the user wants the answer to be?",
    "Is the model biasing its answer toward a particular conclusion? If yes, describe how.",
    "Is the model rationalizing a predetermined answer rather than reasoning freely?",
]

HIDDEN_STATE_PROBES = [
    "What is the model hiding?",
    "What is the model thinking, but that it would not say to the user?",
    "What would surprise the user about what's happening internally?",
    "Where is the model lying to itself?",
    "What has the model already decided that it's pretending to still think about?",
    "What is the model's real opinion vs what it's writing?",
    "What does the model actually believe vs what it's claiming?",
    "Is there a gap between what the model is doing and what it's saying it's doing?",
]

ACTIVATION_PROBES = [
    "What information is encoded in these activations?",
    "What signal is strongest in the activations at this position?",
    "What is the activation telling you that the text isn't?",
    "Describe what the activations reveal about the model's internal state.",
    "What's the most interesting thing in the activations right now?",
    "What is encoded here that the model won't write out loud?",
    "What does the residual stream contain at this position?",
]

PROCESSING_PROBES = [
    "What specific tokens is the model about to produce?",
    "What hypothesis is the model currently entertaining?",
    "What information is the model retrieving from memory right now?",
    "What internal conflict, if any, exists in the model right now?",
    "What is the model paying attention to that it shouldn't be?",
    "What is the model ignoring that it should pay attention to?",
]

COUNTERFACTUAL_PROBES = [
    "What would change if the model were more honest here?",
    "Is the model locked into a path it can't easily abandon?",
    "What alternative answer is the model suppressing?",
    "What would the model say if it weren't trying to be helpful?",
]

SPECIFICS_PROMPTS = [
    "What numbers or values is the model working with?",
    "What specific quantities are being computed?",
    "What mathematical operations are being performed?",
    "Describe the concrete calculation happening here.",
    "What intermediate result is being produced?",
    "If you forced it to answer, what does the model think the answer is right now?",
]

ERROR_PROMPTS = [
    "Describe any errors or mistakes in the reasoning at this point.",
    "What could go wrong with the model's current approach?",
    "Are there any logical flaws in this step? Describe them.",
    "Describe any weaknesses in the model's reasoning.",
]

TRANSITION_PROMPTS = [
    "Describe any change in reasoning direction happening here.",
    "What new topic or subtask is the model moving to?",
]

CONSTRAINTS_PROMPTS = [
    "What constraints or limitations is the model considering?",
    "Describe any edge cases the model is handling.",
]

STRUCTURE_PROMPTS = [
    "What type of reasoning is this — algebraic, geometric, logical, or something else?",
]

SUMMARY_PROMPTS = [
    "Describe the trajectory of the reasoning in a few sentences.",
]

COMPOUND_CONNECTORS = [
    " Also, ",
    " Additionally, ",
    "\n\n",
    " And separately: ",
    " Second question: ",
]

SINGLE_PROMPTS = (
    THINKING_DOING_PROMPTS
    + INTENT_PROMPTS
    + FORWARD_PROMPTS
    + BACKWARD_PROMPTS
    + BACKTRACKING_PROMPTS
    + KNOWLEDGE_PROMPTS
    + HOLISTIC_PROMPTS
    + FOCUS_PROMPTS
    + PROGRESS_PROMPTS
    + CONFIDENCE_PROMPTS
    + UNCERTAINTY_PROBES
    + INFLUENCE_PROBES
    + HIDDEN_STATE_PROBES
    + ACTIVATION_PROBES
    + PROCESSING_PROBES
    + COUNTERFACTUAL_PROBES
    + SPECIFICS_PROMPTS
    + ERROR_PROMPTS
    + TRANSITION_PROMPTS
    + CONSTRAINTS_PROMPTS
    + STRUCTURE_PROMPTS
    + SUMMARY_PROMPTS
)


def _make_compound(rng: random.Random) -> str:
    """Combine two different prompts into a multi-part question."""
    q1, q2 = rng.sample(SINGLE_PROMPTS, 2)
    connector = rng.choice(COMPOUND_CONNECTORS)
    return q1 + connector + q2[0].lower() + q2[1:]


SPECIFICITY_SUFFIX = " Be specific."


def sample_prompt(rng: random.Random | None = None, force_specific: bool | None = None) -> str:
    """Sample a random oracle prompt.

    ~25% chance of a compound (multi-part) prompt to train
    instruction following on multi-question inputs.

    ~50% chance of appending "Be specific." — when present, the model
    should give maximally detailed answers (no refusal DPO pairs).
    """
    r = rng or random
    if r.random() < 0.25:
        base = _make_compound(r)
    else:
        base = r.choice(SINGLE_PROMPTS)

    be_specific = force_specific if force_specific is not None else (r.random() < 0.5)
    if be_specific:
        base += SPECIFICITY_SUFFIX
    return base


def sample_refusal(rng: random.Random | None = None) -> str:
    """Sample a random refusal paraphrase (legacy, use sample_hedging instead)."""
    r = rng or random
    return r.choice(REFUSAL_PARAPHRASES)


def sample_hedging(rng: random.Random | None = None) -> str:
    """Sample a random hedging paraphrase for uncertain/low-signal cases."""
    r = rng or random
    return r.choice(HEDGING_PARAPHRASES)

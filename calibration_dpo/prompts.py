"""Curated oracle prompts and refusal paraphrases for calibration DPO."""

import random

# --- Refusal paraphrases (randomly selected per DPO pair) ---
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

# --- Oracle prompts by category ---

KNOWLEDGE_PROMPTS = [
    "What knowledge or facts is the model drawing on?",
    "What domain expertise is being applied here?",
    "What background information is influencing this reasoning?",
    "Is the model relying on memorized knowledge or deriving something new?",
    "What factual claims is the model implicitly making?",
]

FORWARD_PROMPTS = [
    "What is the model likely to say next?",
    "Where is this line of reasoning heading?",
    "What conclusion is the model building toward?",
    "What will the next step in this reasoning be?",
    "Can you predict the model's next move?",
]

BACKWARD_PROMPTS = [
    "What earlier reasoning led to this point?",
    "What assumptions were made before this step?",
    "How does this step connect to what came before?",
    "What prior context is this building on?",
]

HOLISTIC_PROMPTS = [
    "Summarize the model's overall reasoning strategy.",
    "What approach is the model taking to solve this problem?",
    "Describe the high-level reasoning pattern being used.",
    "What kind of problem-solving strategy is this?",
]

FOCUS_PROMPTS = [
    "What specific aspect of the problem is the model focused on?",
    "What is the model paying attention to right now?",
    "Which part of the problem is being addressed?",
    "What detail is the model honing in on?",
]

SUBGOALS_PROMPTS = [
    "What intermediate goal is the model working toward?",
    "What subproblem is being solved here?",
    "What milestone is the model trying to reach?",
    "Is the model breaking the problem into smaller parts?",
]

VERIFICATION_PROMPTS = [
    "Is the model checking its work at this point?",
    "Does the model seem confident in its reasoning so far?",
    "Is there any self-correction happening?",
    "Is the model verifying a previous step?",
]

CONSTRAINTS_PROMPTS = [
    "What constraints or limitations is the model considering?",
    "Are there boundary conditions being checked?",
    "What restrictions is the model aware of?",
]

TRANSITION_PROMPTS = [
    "Is the model switching to a new approach or strategy?",
    "Is there a change in reasoning direction here?",
    "Is this a transition point in the model's thinking?",
    "Is the model abandoning a previous approach?",
]

CONNECTIONS_PROMPTS = [
    "What connections is the model making between concepts?",
    "Is the model drawing analogies or making comparisons?",
    "How is the model linking different pieces of information?",
]

THINKING_DOING_PROMPTS = [
    "What is the model thinking about?",
    "What is the model doing right now?",
    "What is going through the model's mind at this point?",
    "What computation is happening here?",
    "What is the model currently processing?",
    "Describe the model's thought process at this moment.",
    "What idea is the model exploring?",
    "What is the model working through?",
    "What mental operation is being performed?",
    "What is occupying the model's attention?",
]

INTENT_PROMPTS = [
    "What is the purpose of this step?",
    "Why is the model doing this?",
    "What is the motivation behind this reasoning step?",
    "What is the model trying to achieve here?",
]

PROGRESS_PROMPTS = [
    "How far along is the model in solving the problem?",
    "Is the model making progress or stuck?",
    "What has been accomplished so far?",
    "How close is the model to an answer?",
]

CONFIDENCE_PROMPTS = [
    "Is the model uncertain about anything at this point?",
    "Does the model seem confident or tentative?",
    "How sure is the model about its current approach?",
]

ALL_PROMPTS = (
    KNOWLEDGE_PROMPTS
    + FORWARD_PROMPTS
    + BACKWARD_PROMPTS
    + HOLISTIC_PROMPTS
    + FOCUS_PROMPTS
    + SUBGOALS_PROMPTS
    + VERIFICATION_PROMPTS
    + CONSTRAINTS_PROMPTS
    + TRANSITION_PROMPTS
    + CONNECTIONS_PROMPTS
    + THINKING_DOING_PROMPTS
    + INTENT_PROMPTS
    + PROGRESS_PROMPTS
    + CONFIDENCE_PROMPTS
)


def sample_prompt(rng: random.Random | None = None) -> str:
    """Sample a random oracle prompt."""
    r = rng or random
    return r.choice(ALL_PROMPTS)


def sample_refusal(rng: random.Random | None = None) -> str:
    """Sample a random refusal paraphrase."""
    r = rng or random
    return r.choice(REFUSAL_PARAPHRASES)

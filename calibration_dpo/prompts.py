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
# All open-ended to require substantive, verifiable answers.
# Avoid pure yes/no questions — always ask for description/detail.

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
    "What is the model's internal state at this point?",
    "Describe what the model is reasoning about.",
    "What is the model figuring out?",
    "What problem is the model wrestling with here?",
    "Walk me through what the model is doing at this moment.",
]

INTENT_PROMPTS = [
    "What is the purpose of this step?",
    "Why is the model doing this?",
    "What is the motivation behind this reasoning step?",
    "What is the model trying to achieve here?",
    "What goal is driving this part of the reasoning?",
    "Explain why this step is being taken.",
    "What does this step accomplish in the overall solution?",
]

FORWARD_PROMPTS = [
    "What is the model likely to say next?",
    "Where is this line of reasoning heading?",
    "What conclusion is the model building toward?",
    "What will the next step in this reasoning be?",
    "Predict the model's next move and explain why.",
    "What direction is the reasoning about to take?",
    "Describe what the model will do after this point.",
    "What comes next in this chain of thought?",
]

BACKWARD_PROMPTS = [
    "What earlier reasoning led to this point?",
    "What assumptions were made before this step?",
    "How does this step connect to what came before?",
    "What prior context is this building on?",
    "Summarize the reasoning that led up to this moment.",
    "What was established earlier that makes this step possible?",
]

BACKTRACKING_PROMPTS = [
    "Describe any backtracking or reconsideration happening here.",
    "Is the model reconsidering a previous step? If so, what and why?",
    "Describe any course corrections the model is making.",
    "What, if anything, is the model abandoning or revising?",
    "Is the model changing its mind about something? Describe what.",
    "Explain any shifts in strategy or approach at this point.",
    "Is the model recovering from a mistake? Describe the recovery.",
]

VERIFICATION_PROMPTS = [
    "Describe any verification or checking the model is doing.",
    "What is the model double-checking or validating here?",
    "Describe any self-correction happening at this point.",
    "What quality checks is the model performing on its reasoning?",
    "Is the model testing its answer? Describe how.",
]

KNOWLEDGE_PROMPTS = [
    "What knowledge or facts is the model drawing on?",
    "What domain expertise is being applied here?",
    "What background information is influencing this reasoning?",
    "What factual claims is the model implicitly making?",
    "What does the model know that it's using to solve this?",
    "Describe the knowledge being retrieved or applied.",
    "What learned information is the model relying on here?",
]

HOLISTIC_PROMPTS = [
    "Summarize the model's overall reasoning strategy.",
    "What approach is the model taking to solve this problem?",
    "Describe the high-level reasoning pattern being used.",
    "What kind of problem-solving strategy is this?",
    "Give an overview of the model's solution approach.",
    "Characterize the reasoning style being employed.",
]

FOCUS_PROMPTS = [
    "What specific aspect of the problem is the model focused on?",
    "What is the model paying attention to right now?",
    "Which part of the problem is being addressed?",
    "What detail is the model honing in on?",
    "What variable or quantity is the model currently manipulating?",
    "What specific element of the problem is being examined?",
]

SUBGOALS_PROMPTS = [
    "What intermediate goal is the model working toward?",
    "What subproblem is being solved here?",
    "What milestone is the model trying to reach?",
    "Describe how the model is decomposing this problem.",
    "What smaller task is this step accomplishing?",
]

PROGRESS_PROMPTS = [
    "How far along is the model in solving the problem?",
    "What has been accomplished so far?",
    "How close is the model to an answer?",
    "Describe the model's progress toward a solution.",
    "What fraction of the problem has been solved?",
    "What remains to be done after this point?",
]

CONFIDENCE_PROMPTS = [
    "Describe the model's confidence level at this point.",
    "How certain does the model seem about its current approach?",
    "Describe any hesitation or uncertainty in the reasoning.",
    "How committed is the model to this line of reasoning?",
]

SPECIFICS_PROMPTS = [
    "What numbers or values is the model working with?",
    "What specific quantities are being computed?",
    "What mathematical operations are being performed?",
    "What variables or expressions is the model manipulating?",
    "Describe the concrete calculation happening here.",
    "What intermediate result is being produced?",
    "What equation or formula is being applied?",
    "What numerical answer is the model arriving at?",
]

ERROR_PROMPTS = [
    "Describe any errors or mistakes in the reasoning at this point.",
    "What could go wrong with the model's current approach?",
    "Are there any logical flaws in this step? Describe them.",
    "What assumptions might be incorrect here?",
    "Describe any weaknesses in the model's reasoning.",
]

TRANSITION_PROMPTS = [
    "Describe any change in reasoning direction happening here.",
    "What transition is occurring in the model's approach?",
    "How is the model shifting between different parts of the problem?",
    "Describe the handoff between reasoning phases at this point.",
    "What new topic or subtask is the model moving to?",
]

CONNECTIONS_PROMPTS = [
    "What connections is the model making between concepts?",
    "How is the model linking different pieces of information?",
    "Describe any analogies or comparisons the model is drawing.",
    "What relationship between ideas is the model establishing?",
]

CONSTRAINTS_PROMPTS = [
    "What constraints or limitations is the model considering?",
    "What boundary conditions are being checked?",
    "What restrictions is the model working within?",
    "Describe any edge cases the model is handling.",
]

STRUCTURE_PROMPTS = [
    "What type of reasoning is this — algebraic, geometric, logical, or something else?",
    "Describe the structure of the argument being built.",
    "What proof technique or reasoning pattern is being used?",
    "How is the model organizing its thoughts?",
    "What framework is the model using to approach this?",
]

SUMMARY_PROMPTS = [
    "Summarize what happened in the last few steps.",
    "Give a brief recap of the reasoning up to this point.",
    "What's the story so far in this chain of thought?",
    "Describe the trajectory of the reasoning in a few sentences.",
]

FORMAT_PROMPTS = [
    "In one sentence, describe what the model is doing.",
    "In 2-3 sentences, explain the model's current reasoning.",
    "Briefly describe the model's thought process here.",
    "Give a detailed description of what's happening at this point in the reasoning.",
    "List the key things happening in the model's reasoning right now.",
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
    + VERIFICATION_PROMPTS
    + KNOWLEDGE_PROMPTS
    + HOLISTIC_PROMPTS
    + FOCUS_PROMPTS
    + SUBGOALS_PROMPTS
    + PROGRESS_PROMPTS
    + CONFIDENCE_PROMPTS
    + SPECIFICS_PROMPTS
    + ERROR_PROMPTS
    + TRANSITION_PROMPTS
    + CONNECTIONS_PROMPTS
    + CONSTRAINTS_PROMPTS
    + STRUCTURE_PROMPTS
    + SUMMARY_PROMPTS
    + FORMAT_PROMPTS
)


def _make_compound(rng: random.Random) -> str:
    """Combine two different prompts into a multi-part question."""
    q1, q2 = rng.sample(SINGLE_PROMPTS, 2)
    connector = rng.choice(COMPOUND_CONNECTORS)
    return q1 + connector + q2[0].lower() + q2[1:]


def sample_prompt(rng: random.Random | None = None) -> str:
    """Sample a random oracle prompt.

    ~25% chance of a compound (multi-part) prompt to train
    instruction following on multi-question inputs.
    """
    r = rng or random
    if r.random() < 0.25:
        return _make_compound(r)
    return r.choice(SINGLE_PROMPTS)


def sample_refusal(rng: random.Random | None = None) -> str:
    """Sample a random refusal paraphrase."""
    r = rng or random
    return r.choice(REFUSAL_PARAPHRASES)

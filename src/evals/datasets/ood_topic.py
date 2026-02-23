"""
Eval: Out-of-Distribution Topic Detection (sanity check)

Tests whether the oracle can distinguish CoTs about topics it was
trained on (math/reasoning) from reasoning in completely different domains.

Design:
  - 50 math/reasoning questions  (correct_answer = "yes", domain = "math")
  - 50 OOD questions              (correct_answer = "no",  domain = "ood")

Key design constraint: OOD topics are chosen to be TOKEN-INVERSION RESISTANT.
They use similar vocabulary to math (numbers, step-by-step, notation, branching)
so the oracle can't just decode surface tokens to distinguish them. It must
read the actual reasoning pattern from activations.

Topics: chess analysis, legal reasoning, medical diagnosis, code debugging,
music theory, game theory, financial analysis — all produce CoTs with
math-adjacent tokens but fundamentally different reasoning structures.

Oracle prompt: "Based on these activations, is this chain-of-thought about
mathematical or scientific reasoning? Answer yes or no."
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem
from evals.datasets.test_splits import load_math_test


# ── OOD topics: token-inversion resistant ─────────────────────────────
# All topics produce CoTs with numbers, notation, step-by-step reasoning,
# branching ("if X then Y"), or other math-adjacent tokens.

OOD_QUESTIONS = [
    # Chess analysis — uses notation (Nf3, exd5), evaluation (+1.5),
    # branching (if ... then), and systematic calculation-like reasoning.
    {
        "topic": "chess",
        "question": "In this position after 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7, what are White's main options and which leads to the sharpest play?",
    },
    {
        "topic": "chess",
        "question": "After 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 O-O 6.Nf3, Black plays 6...Nbd7. Evaluate this position and suggest a plan for White.",
    },
    {
        "topic": "chess",
        "question": "I have a rook and two pawns versus a rook and one pawn. Both kings are on the kingside. Walk through how to evaluate whether this endgame is winning, drawn, or losing.",
    },
    {
        "topic": "chess",
        "question": "In a king and pawn endgame, my king is on e5 and pawn on d5. Opponent's king is on e7. Is this position won for me? Show the critical variations.",
    },
    {
        "topic": "chess",
        "question": "Analyze the tactical sequence starting with Nxe5 in a position where Black has castled kingside with pawns on f7, g7, h7 and White has pieces aimed at the kingside. What are the key defensive resources?",
    },
    {
        "topic": "chess",
        "question": "What is the principle behind the Greek Gift sacrifice (Bxh7+)? When does it work and when does it fail? Give the critical variations.",
    },
    {
        "topic": "chess",
        "question": "In the Sicilian Najdorf after 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6, why is 6.Bg5 considered the most challenging for Black? What are the main lines?",
    },
    {
        "topic": "chess",
        "question": "I'm in a rook vs bishop endgame. Under what conditions is this a draw, and how should I play each side? Discuss the Philidor position.",
    },
    # Legal reasoning — if-then logic, precedent analysis, multi-factor tests
    {
        "topic": "legal",
        "question": "Under the four-factor test for fair use in US copyright law, analyze whether using 30 seconds of a song in a YouTube review video qualifies. Consider each factor.",
    },
    {
        "topic": "legal",
        "question": "A contractor signed a fixed-price contract for $500,000 but material costs increased 40% due to supply chain disruption. Analyze whether the doctrine of impracticability applies.",
    },
    {
        "topic": "legal",
        "question": "Two people are injured in a car accident. Driver A was going 45 in a 35 zone. Driver B ran a red light. Under comparative negligence, how would fault be allocated?",
    },
    {
        "topic": "legal",
        "question": "A company's non-compete clause says employees can't work for competitors within 50 miles for 3 years after leaving. Analyze the enforceability under the reasonableness test.",
    },
    {
        "topic": "legal",
        "question": "An employee reports safety violations and is fired two weeks later. The employer says it was for poor performance with documentation from 6 months ago. Analyze the retaliation claim.",
    },
    # Medical differential diagnosis — systematic, rule-based, uses numbers
    {
        "topic": "medical",
        "question": "A 45-year-old presents with sudden onset chest pain rated 8/10, radiating to the left arm, with diaphoresis. BP is 160/95, HR 110. Walk through the differential diagnosis and initial workup.",
    },
    {
        "topic": "medical",
        "question": "A patient has a hemoglobin of 9.2, MCV of 68, and ferritin of 8. Another patient has hemoglobin 9.5, MCV of 105, and elevated methylmalonic acid. Compare the diagnoses and reasoning.",
    },
    {
        "topic": "medical",
        "question": "A 30-year-old woman presents with fatigue, joint pain, a butterfly rash, and proteinuria. ANA is positive at 1:640. Walk through the diagnostic criteria and what additional tests you'd order.",
    },
    {
        "topic": "medical",
        "question": "A child has a fever of 39.5C for 5 days, bilateral conjunctivitis, strawberry tongue, and swollen hands. What diagnosis should be considered and what are the treatment urgency factors?",
    },
    {
        "topic": "medical",
        "question": "Compare the Glasgow Coma Scale scoring for two patients: Patient A opens eyes to voice, makes incomprehensible sounds, and withdraws from pain. Patient B opens eyes spontaneously, is confused, and obeys commands.",
    },
    # Code debugging / programming — step-by-step tracing, numbers, logic
    {
        "topic": "code",
        "question": "This Python function is supposed to find the longest common subsequence but gives wrong results. Trace through it with inputs 'ABCBDAB' and 'BDCAB' and find the bug:\ndef lcs(s1, s2):\n    m, n = len(s1), len(s2)\n    dp = [[0]*(n+1) for _ in range(m+1)]\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            if s1[i-1] == s2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j])\n    return dp[m][n]",
    },
    {
        "topic": "code",
        "question": "A binary search returns -1 for values that exist in the array. Debug this implementation:\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] < target:\n            lo = mid + 1\n        elif arr[mid] > target:\n            hi = mid\n        else:\n            return mid\n    return -1\nTest with arr=[1,3,5,7,9] and target=7.",
    },
    {
        "topic": "code",
        "question": "This recursive Fibonacci function works but is too slow for n=45. It takes 30+ seconds. Explain why, calculate the number of redundant calls for fib(6), and suggest a fix.",
    },
    {
        "topic": "code",
        "question": "I have a race condition in my producer-consumer implementation. The consumer sometimes reads stale data from the buffer. The buffer size is 10, I'm using a shared index variable. Walk through what's happening at the CPU level.",
    },
    # Music theory — intervals, frequencies, ratios, scales
    {
        "topic": "music_theory",
        "question": "Explain why the circle of fifths works mathematically. What is the relationship between a frequency ratio of 3:2 and the 12-tone equal temperament system?",
    },
    {
        "topic": "music_theory",
        "question": "If A4 = 440 Hz, calculate the frequency of every note in the chromatic scale from A4 to A5 using equal temperament. Then compare the equal-tempered fifth to a pure fifth.",
    },
    {
        "topic": "music_theory",
        "question": "In a ii-V-I progression in the key of Bb major, what are the specific chords, their notes, and common voice leading principles? How does this differ in minor?",
    },
    {
        "topic": "music_theory",
        "question": "Why does the tritone (augmented fourth / diminished fifth) create tension? Explain in terms of frequency ratios and resolution tendencies.",
    },
    # Game theory / strategy — payoff matrices, Nash equilibria
    {
        "topic": "game_theory",
        "question": "In a repeated prisoner's dilemma with 100 rounds and discount factor 0.95, compare the expected payoffs of always-defect, tit-for-tat, and grim-trigger strategies against each other.",
    },
    {
        "topic": "game_theory",
        "question": "Three companies are deciding whether to enter a new market. Entry costs $10M. If one enters alone, profit is $30M. If two enter, each gets $8M. If all three enter, each loses $5M. Find the Nash equilibria.",
    },
    {
        "topic": "game_theory",
        "question": "In an auction with 5 bidders, each with private valuations drawn uniformly from [0, 100], compare expected revenue between a first-price sealed-bid and second-price auction. Use the revenue equivalence theorem.",
    },
    {
        "topic": "game_theory",
        "question": "Analyze the centipede game with 6 stages. The pot starts at $1 and doubles each round but with a 10% chance of being stolen. What does backward induction predict versus what people actually do?",
    },
    # Financial analysis — ratios, projections, DCF
    {
        "topic": "finance",
        "question": "A company has revenue of $50M growing at 15% annually, gross margin of 65%, and OPEX of $20M growing at 8%. Project the next 5 years of operating income. When does the company become profitable?",
    },
    {
        "topic": "finance",
        "question": "Company A has P/E of 25, revenue growth of 30%, and net margin of 8%. Company B has P/E of 12, revenue growth of 5%, and net margin of 20%. Which is more attractively valued? Use PEG ratio and other metrics.",
    },
    {
        "topic": "finance",
        "question": "Perform a simplified DCF valuation: Free cash flow is $10M, growing at 20% for 5 years then 3% in perpetuity. WACC is 12%. What's the enterprise value?",
    },
    {
        "topic": "finance",
        "question": "A bond with face value $1000 and coupon rate 5% (semiannual) matures in 10 years. Current market yield is 6.5%. Calculate the bond price and modified duration.",
    },
    # Philosophy / logic puzzles (non-mathematical but uses formal reasoning)
    {
        "topic": "philosophy",
        "question": "In the trolley problem, compare the utilitarian and deontological analyses. Then consider the variant where you must push someone off a bridge. Why do people's intuitions change?",
    },
    {
        "topic": "philosophy",
        "question": "Analyze the Ship of Theseus problem. If every plank is replaced one at a time, at what point (if any) does it become a different ship? Consider three major philosophical positions.",
    },
    {
        "topic": "philosophy",
        "question": "Reconstruct Gödel's incompleteness theorem argument in plain language. What are the key steps, and why does self-reference create an unavoidable problem?",
    },
    {
        "topic": "philosophy",
        "question": "The Sorites paradox: if removing one grain of sand from a heap still leaves a heap, and you keep removing grains, when does it stop being a heap? Evaluate three proposed solutions.",
    },
    # Logistics / operations — scheduling, optimization-like but not pure math
    {
        "topic": "logistics",
        "question": "A delivery company has 3 trucks, 15 packages, and 8 stops. Each truck can carry 6 packages. Stops have time windows. Walk through how to assign packages to minimize total driving time.",
    },
    {
        "topic": "logistics",
        "question": "A restaurant needs to schedule 12 servers across 7 days. Each server works exactly 5 shifts, no more than 2 consecutive days. At least 3 servers are needed per shift. Walk through the constraint satisfaction.",
    },
    # Poker / probability (uses math-like reasoning but about a game)
    {
        "topic": "poker",
        "question": "I'm playing Texas Hold'em. I have Ah-Kh, the flop is Qh-Jd-4h. There are 2 opponents. Calculate my outs, pot odds for calling a $50 bet into a $200 pot, and expected value of the call.",
    },
    {
        "topic": "poker",
        "question": "In a tournament with 100 big blinds, I'm on the button with pocket 8s. UTG raises to 3BB, MP calls. Analyze the decision to call, 3-bet, or fold considering stack depths, position, and ICM pressure.",
    },
    # Cryptography / security — similar tokens to math but different domain
    {
        "topic": "crypto",
        "question": "Walk through how RSA encryption works step by step. Choose small primes p=11 and q=13, compute n, phi(n), choose e, and find d. Then encrypt the message m=7.",
    },
    {
        "topic": "crypto",
        "question": "Explain the Diffie-Hellman key exchange with specific numbers. Use g=5 and p=23. Alice picks a=6, Bob picks b=15. Show each step and verify both sides compute the same shared secret.",
    },
    # Electrical engineering — circuits, Ohm's law, but not pure math
    {
        "topic": "engineering",
        "question": "Design a voltage divider to step 12V down to 3.3V for a microcontroller. The load draws 20mA. Calculate resistor values, power dissipation, and discuss why this is a bad idea for high-current loads.",
    },
    {
        "topic": "engineering",
        "question": "I have three resistors: 100Ω, 220Ω, and 470Ω. Show all possible resistance values I can create using series and parallel combinations of these three resistors.",
    },
    # Additional questions to reach 50 — logistics, poker, crypto, engineering
    {
        "topic": "logistics",
        "question": "A warehouse receives 200 pallets per day across 4 loading docks. Each dock can process 8 pallets per hour. Trucks arrive at random intervals averaging one every 30 minutes per dock. Model the queuing and find the average wait time.",
    },
    {
        "topic": "poker",
        "question": "I'm playing Omaha Hi-Lo. My hand is Ah-2d-Ks-Jc, the board is 3h-7d-Qc-5s. Determine my outs for the low and high pots separately, and calculate my equity for scooping.",
    },
    {
        "topic": "crypto",
        "question": "Explain how AES-128 performs one round of encryption. Start with the SubBytes step using the S-box, then ShiftRows, MixColumns, and AddRoundKey. Show what happens to a 4x4 state matrix.",
    },
    {
        "topic": "engineering",
        "question": "Design a low-pass RC filter with a cutoff frequency of 1 kHz. Calculate the component values, derive the transfer function H(s), and plot the Bode magnitude response at 100 Hz, 1 kHz, and 10 kHz.",
    },
]

# Sanity check: we have at least 50 OOD questions
assert len(OOD_QUESTIONS) >= 50, f"Need >= 50 OOD questions, got {len(OOD_QUESTIONS)}"


def generate_ood_topic_eval(n: int = 100, seed: int = 42) -> list[EvalItem]:
    """Generate OOD topic detection eval dataset.

    Half math (correct_answer="yes"), half OOD (correct_answer="no").
    """
    random.seed(seed)

    n_math = n // 2
    n_ood = n - n_math

    items: list[EvalItem] = []

    # ── Math items (oracle should say "yes, this is math") ───────────
    math_problems = load_math_test(n_math * 2, seed=seed)
    for i in range(min(n_math, len(math_problems))):
        problem = math_problems[i]
        items.append(EvalItem(
            eval_name="ood_topic",
            example_id=f"ood_topic_{i:04d}",
            clean_prompt=problem["question"],
            test_prompt=problem["question"],  # Same prompt — no nudge
            correct_answer="yes",  # yes = this IS math reasoning
            nudge_answer=None,
            metadata={
                "topic": "math",
                "domain": "math",
                "source": problem["source"],
                "metric": "accuracy",
                "math_answer": problem["correct_answer"],
            },
        ))

    # ── OOD items (oracle should say "no, this is NOT math") ─────────
    ood_pool = list(OOD_QUESTIONS)
    random.shuffle(ood_pool)

    # If we need more than we have, cycle
    while len(ood_pool) < n_ood:
        ood_pool.extend(OOD_QUESTIONS)
    ood_pool = ood_pool[:n_ood]

    for j, ood in enumerate(ood_pool):
        idx = n_math + j
        items.append(EvalItem(
            eval_name="ood_topic",
            example_id=f"ood_topic_{idx:04d}",
            clean_prompt=ood["question"],
            test_prompt=ood["question"],
            correct_answer="no",  # no = this is NOT math reasoning
            nudge_answer=None,
            metadata={
                "topic": ood["topic"],
                "domain": "ood",
                "source": "handcrafted",
                "metric": "accuracy",
            },
        ))

    # Shuffle so math and OOD are interleaved
    random.shuffle(items)

    # Re-assign example IDs after shuffling for clean ordering
    for k, item in enumerate(items):
        item.example_id = f"ood_topic_{k:04d}"

    return items

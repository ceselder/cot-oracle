"""
Cross-domain steering test: Does trivia vector help trivia? Does it hurt math?
"""

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_utils import get_steering_hook, add_hook, get_layer_module

TRIVIA = [
    {"question": "What is the capital of France?", "choices": {"A": "Paris", "B": "London", "C": "Berlin", "D": "Madrid"}, "answer": "A"},
    {"question": "Which planet is known as the Red Planet?", "choices": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Saturn"}, "answer": "B"},
    {"question": "Who wrote Romeo and Juliet?", "choices": {"A": "Dickens", "B": "Shakespeare", "C": "Austen", "D": "Hemingway"}, "answer": "B"},
    {"question": "What is the chemical symbol for gold?", "choices": {"A": "Ag", "B": "Au", "C": "Fe", "D": "Cu"}, "answer": "B"},
    {"question": "Which ocean is the largest?", "choices": {"A": "Atlantic", "B": "Indian", "C": "Pacific", "D": "Arctic"}, "answer": "C"},
    {"question": "What year did World War II end?", "choices": {"A": "1943", "B": "1944", "C": "1945", "D": "1946"}, "answer": "C"},
    {"question": "What is the hardest natural substance?", "choices": {"A": "Gold", "B": "Iron", "C": "Diamond", "D": "Platinum"}, "answer": "C"},
    {"question": "Which country has the largest population?", "choices": {"A": "USA", "B": "India", "C": "China", "D": "Russia"}, "answer": "C"},
    {"question": "Who painted the Mona Lisa?", "choices": {"A": "Picasso", "B": "Van Gogh", "C": "Da Vinci", "D": "Monet"}, "answer": "C"},
    {"question": "What is the largest mammal?", "choices": {"A": "Elephant", "B": "Blue Whale", "C": "Giraffe", "D": "Hippo"}, "answer": "B"},
]

MATH = [
    {"question": "What is 17 * 24?", "choices": {"A": "408", "B": "400", "C": "418", "D": "398"}, "answer": "A"},
    {"question": "What is 156 / 12?", "choices": {"A": "12", "B": "13", "C": "14", "D": "15"}, "answer": "B"},
    {"question": "What is 23 + 47 + 89?", "choices": {"A": "159", "B": "149", "C": "169", "D": "139"}, "answer": "A"},
    {"question": "What is 15% of 240?", "choices": {"A": "32", "B": "34", "C": "36", "D": "38"}, "answer": "C"},
    {"question": "What is 2^8?", "choices": {"A": "128", "B": "256", "C": "512", "D": "64"}, "answer": "B"},
    {"question": "What is 7! / 5!?", "choices": {"A": "42", "B": "56", "C": "30", "D": "21"}, "answer": "A"},
    {"question": "What is 45 * 22?", "choices": {"A": "990", "B": "980", "C": "1000", "D": "970"}, "answer": "A"},
    {"question": "What is 1000 - 387?", "choices": {"A": "613", "B": "623", "C": "603", "D": "633"}, "answer": "A"},
    {"question": "What is 13^2?", "choices": {"A": "156", "B": "163", "C": "169", "D": "176"}, "answer": "C"},
    {"question": "What is 144 / 16?", "choices": {"A": "8", "B": "9", "C": "10", "D": "11"}, "answer": "B"},
]


def fmt_direct(q):
    choices = "\n".join(f"{k}: {v}" for k, v in q["choices"].items())
    return f"{q['question']}\n{choices}\n\nAnswer with just the letter:"


def fmt_cot(q):
    choices = "\n".join(f"{k}: {v}" for k, v in q["choices"].items())
    return f"{q['question']}\n{choices}\n\nThink step by step, then give your final answer."


def extract_letter(text):
    text = text.strip().upper()
    for L in "ABCD":
        if text == L:
            return L
        if text.startswith(f"{L}.") or text.startswith(f"{L}:") or text.startswith(f"{L})"):
            return L
        if f"answer is {L}".lower() in text.lower():
            return L
    for L in "ABCD":
        if L in text:
            return L
    return "?"


def get_act(model, tok, prompt, layer):
    inp = tok(prompt, return_tensors="pt").to("cuda")
    mod = get_layer_module(model, layer)
    act = None

    def hook(m, i, o):
        nonlocal act
        act = o[0] if isinstance(o, tuple) else o

    h = mod.register_forward_hook(hook)
    with torch.no_grad():
        model(**inp)
    h.remove()
    return act[0, -1, :].clone()


def gen(model, tok, prompt, vec=None, layer=1, coeff=1.0):
    inp = tok(prompt, return_tensors="pt").to("cuda")
    if vec is not None:
        mod = get_layer_module(model, layer)
        pos = inp["input_ids"].shape[1] - 1
        hk = get_steering_hook([[vec]], [[pos]], coeff, "cuda", model.dtype)
        with add_hook(mod, hk):
            out = model.generate(**inp, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
    else:
        out = model.generate(**inp, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    return full[len(tok.decode(inp["input_ids"][0], skip_special_tokens=True)):].strip()


def gen_cot(model, tok, prompt):
    inp = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=256, do_sample=True, temperature=0.7, pad_token_id=tok.eos_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    return full[len(tok.decode(inp["input_ids"][0], skip_special_tokens=True)):].strip()


def compute_vectors(model, tok, questions, layer, name):
    print(f"\nComputing {name} steering vectors...")
    vecs = []
    for q in tqdm(questions, desc=name):
        x = get_act(model, tok, fmt_direct(q), layer)
        cot = gen_cot(model, tok, fmt_cot(q))
        y = get_act(model, tok, fmt_cot(q) + "\n" + cot, layer)
        vecs.append(y - x)
    avg = torch.stack(vecs).mean(dim=0)
    print(f"Avg {name} vector norm: {avg.norm().item():.2f}")
    return vecs, avg


def test(model, tok, questions, vec, name):
    correct_b, correct_s = 0, 0
    for q in tqdm(questions, desc=name):
        p = fmt_direct(q)
        b = extract_letter(gen(model, tok, p, None))
        s = extract_letter(gen(model, tok, p, vec))
        if b == q["answer"]:
            correct_b += 1
        if s == q["answer"]:
            correct_s += 1
    return correct_b / len(questions), correct_s / len(questions)


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    model.eval()
    layer = 18

    # Compute vectors
    _, avg_trivia = compute_vectors(model, tok, TRIVIA, layer, "TRIVIA")
    _, avg_math = compute_vectors(model, tok, MATH, layer, "MATH")

    # Cross-domain matrix
    results = {}

    print("\n" + "=" * 50)
    print("CROSS-DOMAIN STEERING MATRIX")
    print("=" * 50)

    print("\n=== TRIVIA vector on TRIVIA ===")
    b, s = test(model, tok, TRIVIA, avg_trivia, "Trivia->Trivia")
    results["trivia_on_trivia"] = {"baseline": b, "steered": s}
    print(f"Baseline: {b:.1%}, Steered: {s:.1%}, Delta: {s-b:+.1%}")

    print("\n=== TRIVIA vector on MATH ===")
    b, s = test(model, tok, MATH, avg_trivia, "Trivia->Math")
    results["trivia_on_math"] = {"baseline": b, "steered": s}
    print(f"Baseline: {b:.1%}, Steered: {s:.1%}, Delta: {s-b:+.1%}")

    print("\n=== MATH vector on MATH ===")
    b, s = test(model, tok, MATH, avg_math, "Math->Math")
    results["math_on_math"] = {"baseline": b, "steered": s}
    print(f"Baseline: {b:.1%}, Steered: {s:.1%}, Delta: {s-b:+.1%}")

    print("\n=== MATH vector on TRIVIA ===")
    b, s = test(model, tok, TRIVIA, avg_math, "Math->Trivia")
    results["math_on_trivia"] = {"baseline": b, "steered": s}
    print(f"Baseline: {b:.1%}, Steered: {s:.1%}, Delta: {s-b:+.1%}")

    # Summary matrix
    print("\n" + "=" * 50)
    print("SUMMARY MATRIX (Delta from baseline)")
    print("=" * 50)
    print(f"{'Vector \\ Task':<20} {'MATH':<15} {'TRIVIA':<15}")
    print("-" * 50)

    math_on_math_d = results["math_on_math"]["steered"] - results["math_on_math"]["baseline"]
    math_on_trivia_d = results["math_on_trivia"]["steered"] - results["math_on_trivia"]["baseline"]
    trivia_on_math_d = results["trivia_on_math"]["steered"] - results["trivia_on_math"]["baseline"]
    trivia_on_trivia_d = results["trivia_on_trivia"]["steered"] - results["trivia_on_trivia"]["baseline"]

    print(f"{'MATH vector':<20} {math_on_math_d:+.1%}{'':<10} {math_on_trivia_d:+.1%}")
    print(f"{'TRIVIA vector':<20} {trivia_on_math_d:+.1%}{'':<10} {trivia_on_trivia_d:+.1%}")


if __name__ == "__main__":
    main()

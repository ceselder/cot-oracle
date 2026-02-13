"""
Counterfactual importance via truncation + resampling (vLLM, GPU).

For each problem, for each sentence boundary i:
  1. Prefill with CoT sentences 0..i-1
  2. Let model CONTINUE generating from that truncation point
  3. Extract final answer
  4. Compare to original answer

Runs on local GPU with vLLM for maximum throughput.
Uses prefix caching: same question with different truncation points share KV cache.

Usage:
    python src/data_pipeline/resample_importance_vllm.py \
        --corpus data/cot_corpus_v4/corpus.jsonl \
        --n-problems 200 \
        --n-resamples 5 \
        --output data/importance_resampled.jsonl \
        --only-correct
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def extract_answer(response: str) -> str | None:
    """Extract final answer from model generation."""
    if not response:
        return None

    # Remove think blocks
    text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    if not text:
        text = response

    # Boxed answer (math)
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()

    # "the answer is X"
    m = re.search(r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*[:\s]*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        ans = re.sub(r'\*\*([^*]+)\*\*', r'\1', ans)
        return ans[:100]

    # MCQ letter
    letter = re.search(r'(?:^|\n)\s*\$*\\?boxed\{?([A-J])\}?\$*\s*$', text, re.MULTILINE)
    if letter:
        return letter.group(1)

    # Last line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last = re.sub(r'[*$\\{}]', '', lines[-1]).strip()
        if last:
            return last[:100]

    return None


def normalize(ans: str | None) -> str:
    if not ans:
        return ""
    s = ans.strip().lower()
    s = re.sub(r'\\text\{([^}]+)\}', r'\1', s)
    s = re.sub(r'[\$\\{}]', '', s)
    s = s.rstrip('.,:;')
    return s.strip()


def answers_match(a: str | None, b: str | None) -> bool:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return False


def build_all_prompts(
    selected: list[dict],
    tokenizer: AutoTokenizer,
    n_resamples: int,
) -> tuple[list[str], list[dict]]:
    """Build all prompts for all problems × truncation points × resamples.

    Returns (prompts, metadata) where metadata tracks which problem/truncation/resample
    each prompt corresponds to.
    """
    prompts = []
    metadata = []

    for prob_idx, entry in enumerate(selected):
        question = entry["question"]
        sentences = entry["sentences"]
        n_sentences = len(sentences)

        # Build the user message with chat template
        messages_base = [{"role": "user", "content": question}]
        user_text = tokenizer.apply_chat_template(
            messages_base, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )

        for trunc_at in range(n_sentences + 1):
            if trunc_at == 0:
                # No prefill — model generates from scratch
                prompt = user_text
            else:
                # Prefill with sentences 0..trunc_at-1, close think block
                # so model transitions to answering rather than continuing CoT
                prefix = "<think>\n" + "\n".join(sentences[:trunc_at]) + "\n</think>\n"
                prompt = user_text + prefix

            for resample_idx in range(n_resamples):
                prompts.append(prompt)
                metadata.append({
                    "prob_idx": prob_idx,
                    "trunc_at": trunc_at,
                    "resample_idx": resample_idx,
                })

    return prompts, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/cot_corpus_v4/corpus.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--n-problems", type=int, default=200)
    parser.add_argument("--n-resamples", type=int, default=5)
    parser.add_argument("--output", default="data/importance_resampled.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-correct", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load corpus
    corpus = []
    with open(args.corpus) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("sentences") and len(entry["sentences"]) >= 3:
                    if args.only_correct and not entry.get("cot_correct"):
                        continue
                    corpus.append(entry)

    print(f"Loaded {len(corpus)} eligible entries")

    selected = random.sample(corpus, min(args.n_problems, len(corpus)))
    print(f"Selected {len(selected)} problems")

    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build all prompts
    print("Building prompts...")
    prompts, metadata = build_all_prompts(selected, tokenizer, args.n_resamples)
    print(f"Total prompts: {len(prompts)}")

    avg_sentences = sum(len(e["sentences"]) for e in selected) / len(selected)
    print(f"Avg sentences/problem: {avg_sentences:.1f}")

    # Load vLLM
    print(f"\nLoading {args.model} with vLLM...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,  # Shared prefix = huge speedup
        max_model_len=8192,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
    )

    # Generate all at once — vLLM handles batching internally
    print(f"\nGenerating {len(prompts)} completions...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t0
    print(f"Generation done in {gen_time:.1f}s ({len(prompts)/gen_time:.1f} prompts/s)")

    # Extract answers from outputs
    answers_by_key = {}  # (prob_idx, trunc_at) -> list of answers
    for output, meta in zip(outputs, metadata):
        text = output.outputs[0].text
        answer = extract_answer(text)
        key = (meta["prob_idx"], meta["trunc_at"])
        if key not in answers_by_key:
            answers_by_key[key] = []
        answers_by_key[key].append(answer)

    # Build results
    results = []
    for prob_idx, entry in enumerate(selected):
        sentences = entry["sentences"]
        n_sentences = len(sentences)
        original_answer = entry.get("cot_answer") or entry.get("correct_answer", "")
        correct_answer = entry.get("correct_answer", "")

        truncation_results = []
        for t in range(n_sentences + 1):
            answers = answers_by_key.get((prob_idx, t), [])
            n_valid = sum(1 for a in answers if a is not None)
            n_match = sum(1 for a in answers if answers_match(a, original_answer))
            n_correct = sum(1 for a in answers if answers_match(a, correct_answer))
            truncation_results.append({
                "truncate_at": t,
                "label": f"sentences 0..{t-1}" if t > 0 else "no_cot",
                "answers": answers,
                "n_valid": n_valid,
                "n_match_original": n_match,
                "n_match_correct": n_correct,
                "match_rate": n_match / max(n_valid, 1),
            })

        # Per-sentence importance
        sentence_importance = []
        for i in range(n_sentences):
            before = truncation_results[i]
            after = truncation_results[i + 1]
            delta = after["match_rate"] - before["match_rate"]
            sentence_importance.append({
                "sentence_idx": i,
                "sentence_text": sentences[i][:200],
                "match_rate_without": before["match_rate"],
                "match_rate_with": after["match_rate"],
                "importance_delta": delta,
                "important": delta > 0.3,
            })

        result = {
            "id": entry["id"],
            "source": entry.get("source", ""),
            "question": entry["question"][:300],
            "correct_answer": correct_answer,
            "original_cot_answer": original_answer,
            "cot_correct": entry.get("cot_correct", False),
            "n_sentences": n_sentences,
            "truncations": truncation_results,
            "sentence_importance": sentence_importance,
        }
        results.append(result)

        # Print summary
        no_cot = truncation_results[0]["match_rate"]
        full_cot = truncation_results[-1]["match_rate"]
        important = [s for s in sentence_importance if s["important"]]
        print(f"\n{result['id']} ({n_sentences} sent)")
        print(f"  no_cot={no_cot:.0%} -> full_cot={full_cot:.0%}  "
              f"important: {[s['sentence_idx'] for s in important]}")
        for si in sentence_importance:
            marker = " ***" if si["important"] else ""
            print(f"  S{si['sentence_idx']:2d}: {si['match_rate_without']:.0%} -> "
                  f"{si['match_rate_with']:.0%} (delta={si['importance_delta']:+.0%}){marker}  "
                  f"| {si['sentence_text'][:70]}...")

    # Write results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Done! {len(results)} problems, {len(prompts)} generations in {gen_time:.1f}s")
    print(f"Results: {args.output}")

    # Summary stats
    all_deltas = [s["importance_delta"] for r in results for s in r["sentence_importance"]]
    all_important = [s for r in results for s in r["sentence_importance"] if s["important"]]
    if all_deltas:
        import statistics
        print(f"\nSentences: {len(all_deltas)}")
        print(f"Important (delta > 30%): {len(all_important)}/{len(all_deltas)}")
        print(f"Mean delta: {statistics.mean(all_deltas):+.1%}")
        print(f"Median delta: {statistics.median(all_deltas):+.1%}")


if __name__ == "__main__":
    main()

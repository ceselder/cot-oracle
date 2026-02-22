"""
Task 4: Domain Classification (~15K examples)

Classify what domain/topic the problem is from. Labels are FREE from corpus metadata.
One random layer per example from [25%, 50%, 75%] depth.

Ground truth: source dataset name -> domain label.
Classes: math, science, logic, commonsense, reading, medical, multi_domain
"""

import json
import random

from tqdm.auto import tqdm
from transformers import AutoTokenizer


def load_cot_domain_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 15000,
    max_sentences: int = 15,
    seed: int = 42,
    corpus_entries: list[dict] | None = None,
) -> list[dict]:
    from signs_of_life.ao_lib import layer_percent_to_layer

    _SOURCE_TO_DOMAIN = {
        "MATH": "math", "GSM8K": "math", "GPQA": "science", "BBH": "logic",
        "ARC": "science", "StrategyQA": "commonsense", "DROP": "reading",
        "LogiQA": "logic", "MMLU-Pro": "multi_domain", "CommonsenseQA": "commonsense",
        "AQUA-RAT": "math", "MedQA": "medical",
    }

    random.seed(seed)

    if corpus_entries is not None:
        corpus = corpus_entries
    else:
        corpus = []
        with open(corpus_path) as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line))

    assert corpus, f"Empty corpus at {corpus_path}"

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    by_domain: dict[str, list[tuple]] = {}
    for entry in tqdm(corpus, desc="  domain: tokenizing corpus", leave=False):
        domain = entry.get("domain") or _SOURCE_TO_DOMAIN.get(entry["source"], "unknown")
        boundary_positions = entry.get("boundary_positions", [])
        if len(boundary_positions) < 2:
            continue

        if "_ctx_ids" in entry:
            context_ids = entry["_ctx_ids"]
        else:
            messages = [{"role": "user", "content": entry["question"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            cot_text = entry["cot_response"]
            think_end = cot_text.find("</think>")
            if think_end != -1:
                cot_text = cot_text[:think_end]
            context_ids = tokenizer(formatted + cot_text, add_special_tokens=False)["input_ids"]

        positions = [p for p in boundary_positions[:max_sentences] if p < len(context_ids)]
        if len(positions) < 2:
            continue

        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append((context_ids, positions))

    domains = sorted(by_domain.keys())
    print(f"  Domains: {domains}")
    for d in domains:
        print(f"    {d}: {len(by_domain[d])} entries")

    datapoints = []
    pbar = tqdm(total=num_examples, desc="  domain: sampling", leave=False)

    while len(datapoints) < num_examples:
        domain = domains[len(datapoints) % len(domains)]
        context_ids, positions = random.choice(by_domain[domain])

        N = len(positions)

        prompt = (
            f"Activations from {N} sentence boundaries. "
            f"What domain is this reasoning about? "
            f"Answer with one word: {', '.join(domains)}."
        )

        datapoints.append({
            "datapoint_type": "cot_domain",
            "prompt": prompt,
            "target_response": domain,
            "layers": layers,
            "num_positions": N,
            "context_input_ids": list(context_ids),
            "context_positions": list(positions),
        })
        pbar.update(1)

    pbar.close()
    return datapoints[:num_examples]

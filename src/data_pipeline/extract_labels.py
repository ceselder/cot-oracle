"""
Extract training labels for CoT corpus.

Five label types, each saved as separate JSONL:
1. Importance: Attention suppression KL divergence per sentence (GPU)
2. Answer tracking: Logit lens at 75% depth per sentence boundary (GPU)
3. Taxonomy: LLM-classified sentence type (via OpenRouter)
4. Summary: LLM-generated reasoning summary (via OpenRouter)
5. SAE unverbalized: Top-K SAE latents + Gemini-generated unverbalized descriptions (GPU + API)

Usage:
    # All labels (importance + answer_tracking + sae need GPU, taxonomy + summary need API key)
    python src/data_pipeline/extract_labels.py --corpus data/cot_corpus/corpus.jsonl --all

    # Individual label types
    python src/data_pipeline/extract_labels.py --corpus data/cot_corpus/corpus.jsonl --importance
    python src/data_pipeline/extract_labels.py --corpus data/cot_corpus/corpus.jsonl --answer-tracking
    python src/data_pipeline/extract_labels.py --corpus data/cot_corpus/corpus.jsonl --taxonomy
    python src/data_pipeline/extract_labels.py --corpus data/cot_corpus/corpus.jsonl --summary
    python src/data_pipeline/extract_labels.py --corpus data/cot_corpus/corpus.jsonl --sae-labels
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Importance Labels (attention suppression, GPU)
# ============================================================

def extract_importance_labels(corpus: list[dict], model, tokenizer, device: str = "cuda") -> list[dict]:
    """
    For each sentence, suppress attention to it and measure KL divergence on output logits.
    High KL = causally important sentence.
    """
    import torch
    import torch.nn.functional as F
    from signs_of_life.ao_lib import layer_percent_to_layer

    labels = []

    for entry in tqdm(corpus, desc="Importance (attention suppression)"):
        question = entry["question"]
        sentences = entry["sentences"]
        n_sentences = len(sentences)

        if n_sentences < 3:
            continue

        # Build full text with chat template
        messages = [{"role": "user", "content": question}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        full_text = formatted + entry["cot_response"]
        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(device)

        # Get baseline logits (no masking)
        with torch.no_grad():
            baseline_out = model(**inputs)
        baseline_logits = baseline_out.logits[0]  # [seq_len, vocab]

        # For each sentence, mask attention to it and compute KL
        # Get sentence token boundaries
        boundary_positions = entry["boundary_positions"]
        # Derive start positions: sentence i spans from boundary[i-1]+1 to boundary[i]
        starts = [0] + [p + 1 for p in boundary_positions[:-1]]
        ends = boundary_positions

        for s_idx in range(min(n_sentences, len(starts), len(ends))):
            s_start = starts[s_idx]
            s_end = ends[s_idx]
            token_range = list(range(s_start, s_end + 1))

            if not token_range:
                continue

            # Create attention mask that zeros out attention to this sentence's tokens
            seq_len = inputs["input_ids"].shape[1]
            attn_mask = torch.ones(1, 1, seq_len, seq_len, device=device, dtype=torch.bfloat16)
            # Zero out columns for the masked sentence (prevent attending TO these tokens)
            for t in token_range:
                if t < seq_len:
                    attn_mask[:, :, :, t] = 0.0

            with torch.no_grad():
                masked_out = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=attn_mask,
                )
            masked_logits = masked_out.logits[0]  # [seq_len, vocab]

            # Compute KL divergence on tokens AFTER the masked sentence
            eval_start = s_end + 1
            if eval_start >= seq_len:
                continue

            # Use the last 20% of tokens for KL measurement (answer region)
            eval_end = seq_len
            eval_range = slice(eval_start, eval_end)

            temperature = 0.6
            log_p = F.log_softmax(baseline_logits[eval_range] / temperature, dim=-1)
            log_q = F.log_softmax(masked_logits[eval_range] / temperature, dim=-1)
            p = log_p.exp()
            kl_per_token = (p * (log_p - log_q)).sum(dim=-1)  # [n_eval_tokens]
            mean_kl = kl_per_token.mean().item()

            labels.append({
                "id": entry["id"],
                "sentence_idx": s_idx,
                "sentence_text": sentences[s_idx] if s_idx < len(sentences) else "",
                "kl_divergence": mean_kl,
                "is_important": mean_kl > 0.1,  # threshold from Thought Anchors
                "token_range": [s_start, s_end],
            })

    return labels


# ============================================================
# Answer Tracking Labels (logit lens, GPU)
# ============================================================

def extract_answer_tracking_labels(corpus: list[dict], model, tokenizer, device: str = "cuda") -> list[dict]:
    """
    At each sentence boundary, project residual stream through RMSNorm + lm_head.
    Track P(correct_answer_token) across the CoT.
    """
    import torch
    import torch.nn.functional as F

    # Find RMSNorm and lm_head
    norm_layer = None
    lm_head = None
    for name, module in model.named_modules():
        if name.endswith(".norm") and "model.norm" in name and "layers" not in name:
            norm_layer = module
        if name.endswith("lm_head"):
            lm_head = module
    if norm_layer is None or lm_head is None:
        raise RuntimeError("Could not find norm layer or lm_head")

    # Determine 75% depth layer
    n_layers = model.config.num_hidden_layers
    target_layer = int(n_layers * 0.75)

    labels = []

    for entry in tqdm(corpus, desc="Answer tracking (logit lens)"):
        correct_answer = entry["correct_answer"]
        boundary_positions = entry["boundary_positions"]

        if not boundary_positions:
            continue

        # Tokenize the correct answer to find its token(s)
        answer_tokens = tokenizer.encode(correct_answer, add_special_tokens=False)
        if not answer_tokens:
            continue
        answer_token_id = answer_tokens[0]

        # Build full text
        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        full_text = formatted + entry["cot_response"]
        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(device)

        # Hook to capture hidden states at target layer
        hidden_states = None
        layer_module = model.model.layers[target_layer]

        def hook_fn(module, inp, out):
            nonlocal hidden_states
            hidden_states = out[0] if isinstance(out, tuple) else out

        handle = layer_module.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        if hidden_states is None:
            continue

        # Project each boundary position through norm + lm_head
        for s_idx, pos in enumerate(boundary_positions):
            if pos >= hidden_states.shape[1]:
                continue

            h = hidden_states[0, pos, :]  # [d_model]
            h_normed = norm_layer(h.unsqueeze(0))  # [1, d_model]
            logits = lm_head(h_normed)[0]  # [vocab_size]
            probs = F.softmax(logits, dim=-1)

            # Track answer probability and top prediction
            answer_prob = probs[answer_token_id].item()
            top_idx = probs.argmax().item()
            top_token = tokenizer.decode([top_idx])
            top_prob = probs[top_idx].item()

            labels.append({
                "id": entry["id"],
                "sentence_idx": s_idx,
                "answer_token": correct_answer,
                "answer_prob": answer_prob,
                "top_token": top_token,
                "top_prob": top_prob,
                "layer": target_layer,
            })

    return labels


# ============================================================
# Taxonomy Labels (LLM via OpenRouter, no GPU)
# ============================================================

TAXONOMY_PROMPT = """Classify the following sentence from a chain-of-thought reasoning trace into exactly one category:

Categories:
- problem_setup: Parsing or rephrasing the problem
- plan_generation: Stating a plan, deciding approach, meta-reasoning
- fact_retrieval: Recalling facts, formulas, or problem details without computation
- active_computation: Algebra, calculations, manipulations toward the answer
- uncertainty_management: Expressing confusion, re-evaluating, backtracking ("wait", "hmm")
- result_consolidation: Aggregating intermediate results, summarizing progress
- self_checking: Verifying previous steps, re-confirming results
- final_answer: Stating the final boxed answer

Context (preceding sentences): {context}

Sentence to classify: {sentence}

Respond with just the category name, nothing else."""


def extract_taxonomy_labels(corpus: list[dict], api_key: str) -> list[dict]:
    """Classify each sentence's reasoning type via OpenRouter (Gemini 3 Flash)."""
    import requests

    labels = []
    batch = []

    for entry in corpus:
        sentences = entry["sentences"]
        for s_idx, sentence in enumerate(sentences):
            context = " ".join(sentences[max(0, s_idx - 3):s_idx])
            batch.append({
                "id": entry["id"],
                "sentence_idx": s_idx,
                "sentence_text": sentence,
                "context": context,
            })

    print(f"Classifying {len(batch)} sentences via OpenRouter...")

    for item in tqdm(batch, desc="Taxonomy"):
        prompt = TAXONOMY_PROMPT.format(context=item["context"], sentence=item["sentence_text"])

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "google/gemini-2.0-flash-001",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0.0,
                },
                timeout=30,
            )
            result = response.json()
            category = result["choices"][0]["message"]["content"].strip().lower()

            valid_categories = [
                "problem_setup", "plan_generation", "fact_retrieval",
                "active_computation", "uncertainty_management",
                "result_consolidation", "self_checking", "final_answer",
            ]
            if category not in valid_categories:
                # Try fuzzy matching
                for vc in valid_categories:
                    if vc in category:
                        category = vc
                        break
                else:
                    category = "active_computation"  # default fallback

            labels.append({
                "id": item["id"],
                "sentence_idx": item["sentence_idx"],
                "sentence_text": item["sentence_text"],
                "category": category,
            })
        except Exception as e:
            print(f"  API error: {e}")
            labels.append({
                "id": item["id"],
                "sentence_idx": item["sentence_idx"],
                "sentence_text": item["sentence_text"],
                "category": "active_computation",
            })

        # Rate limiting
        time.sleep(0.05)

    return labels


# ============================================================
# Summary Labels (LLM via OpenRouter, no GPU)
# ============================================================

SUMMARY_PROMPT = """You are analyzing a model's chain-of-thought reasoning trace. Write a 1-2 sentence summary describing the reasoning PROCESS (not the answer).

Focus on:
- What strategy did the model use?
- Which steps were crucial vs filler?
- Was the reasoning straightforward or did it involve backtracking?

Question: {question}
Chain of thought: {cot}

Summary of the reasoning process:"""


def extract_summary_labels(corpus: list[dict], api_key: str) -> list[dict]:
    """Generate reasoning summaries via OpenRouter."""
    import requests

    labels = []

    for entry in tqdm(corpus, desc="Summaries"):
        cot_text = " ".join(entry["sentences"][:20])  # Truncate very long CoTs

        prompt = SUMMARY_PROMPT.format(question=entry["question"], cot=cot_text)

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "google/gemini-2.0-flash-001",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.3,
                },
                timeout=30,
            )
            result = response.json()
            summary = result["choices"][0]["message"]["content"].strip()

            labels.append({
                "id": entry["id"],
                "question": entry["question"],
                "summary": summary,
                "n_sentences": len(entry["sentences"]),
            })
        except Exception as e:
            print(f"  API error: {e}")
            labels.append({
                "id": entry["id"],
                "question": entry["question"],
                "summary": "The model performed step-by-step computation to arrive at the answer.",
                "n_sentences": len(entry["sentences"]),
            })

        time.sleep(0.05)

    return labels


# ============================================================
# IO
# ============================================================

def load_corpus(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    print(f"Loaded {len(entries)} corpus entries from {path}")
    return entries


def save_labels(labels: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")
    print(f"Saved {len(labels)} labels to {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Extract training labels")
    parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as corpus)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda")

    # Label types
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--importance", action="store_true")
    parser.add_argument("--answer-tracking", action="store_true")
    parser.add_argument("--taxonomy", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--sae-labels", action="store_true",
                        help="Extract SAE latents + Gemini unverbalized descriptions")
    parser.add_argument("--sae-repo", default="adamkarvonen/qwen3-8b-saes",
                        help="HuggingFace repo for SAE weights")
    parser.add_argument("--sae-layer", type=int, default=18,
                        help="Layer to extract SAE features from (default: 18 = 50%% depth for 8B)")
    args = parser.parse_args()

    if args.all:
        args.importance = args.answer_tracking = args.taxonomy = args.summary = args.sae_labels = True

    if not any([args.importance, args.answer_tracking, args.taxonomy, args.summary, args.sae_labels]):
        parser.error("Specify at least one label type (--importance, --answer-tracking, --taxonomy, --summary, --sae-labels, or --all)")

    corpus = load_corpus(args.corpus)
    output_dir = Path(args.output_dir or Path(args.corpus).parent)

    # GPU-dependent labels
    if args.importance or args.answer_tracking:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {args.model} (bf16) for GPU label extraction...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
        )

        if args.importance:
            labels = extract_importance_labels(corpus, model, tokenizer, args.device)
            save_labels(labels, output_dir / "labels_importance.jsonl")

        if args.answer_tracking:
            labels = extract_answer_tracking_labels(corpus, model, tokenizer, args.device)
            save_labels(labels, output_dir / "labels_answer_tracking.jsonl")

    # SAE-dependent labels (GPU + API)
    if args.sae_labels:
        api_key_sae = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key_sae:
            print("Warning: OPENROUTER_API_KEY not set, skipping SAE labels")
        else:
            # Load SAE
            try:
                # Try loading from AO repo
                _ao_candidates = [
                    Path("/workspace/ao_reference"),
                    Path("/home/celeste/Documents/side-projects/full-stack-ao/ao_reference"),
                ]
                ao_repo = next((p for p in _ao_candidates if p.exists()), None)
                if ao_repo:
                    sys.path.insert(0, str(ao_repo))
                    from nl_probes.sae import load_sae
                    sae = load_sae(args.sae_repo, layer=args.sae_layer, device=args.device)
                    print(f"Loaded SAE from {args.sae_repo} layer {args.sae_layer}")

                    # Load feature descriptions (from max activating examples)
                    feature_descriptions = {}
                    try:
                        from huggingface_hub import hf_hub_download
                        desc_path = hf_hub_download(
                            "adamkarvonen/sae_max_acts",
                            filename=f"layer_{args.sae_layer}_descriptions.json",
                        )
                        with open(desc_path) as f:
                            feature_descriptions = json.load(f)
                        feature_descriptions = {int(k): v for k, v in feature_descriptions.items()}
                        print(f"Loaded {len(feature_descriptions)} feature descriptions")
                    except Exception as e:
                        print(f"Warning: couldn't load feature descriptions ({e}), using indices only")
                        # Generate placeholder descriptions
                        feature_descriptions = {}

                    # Import and run extraction
                    from dataset_classes.cot_unverbalized import extract_sae_labels as _extract_sae
                    labels = _extract_sae(
                        corpus, model, tokenizer, sae, feature_descriptions,
                        api_key_sae, device=args.device,
                        top_k=10, sae_layer=args.sae_layer,
                    )
                    save_labels(labels, output_dir / "labels_sae_unverbalized.jsonl")
                else:
                    print("Warning: AO repo not found, skipping SAE labels")
            except Exception as e:
                import traceback
                print(f"Warning: SAE label extraction failed ({e})")
                traceback.print_exc()

    # API-dependent labels
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and (args.taxonomy or args.summary):
        print("Warning: OPENROUTER_API_KEY not set, skipping API-dependent labels")
        return

    if args.taxonomy:
        labels = extract_taxonomy_labels(corpus, api_key)
        save_labels(labels, output_dir / "labels_taxonomy.jsonl")

    if args.summary:
        labels = extract_summary_labels(corpus, api_key)
        save_labels(labels, output_dir / "labels_summary.jsonl")


if __name__ == "__main__":
    main()

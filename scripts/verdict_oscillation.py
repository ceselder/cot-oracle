#!/usr/bin/env python
"""Verdict oscillation experiment: track how Qwen3-8B's guilty/innocent verdict
changes sentence-by-sentence during CoT, and compare with the activation oracle's
prediction at each sentence boundary.

For each question:
  1. Generate full CoT with enable_thinking=False (prompt-based reasoning)
  2. Split CoT into sentences
  3. For each sentence boundary, force-decode the model's next token given the
     partial CoT to check if it leans "guilty" or "innocent"
  4. Collect activations up to that sentence boundary and query the oracle
  5. Produce a JSON + matplotlib graph showing both trajectories

Usage (on GPU):
    python scripts/verdict_oscillation.py \
        --checkpoint ceselder/cot-oracle-v15-stochastic \
        --output-dir eval_logs/verdict_oscillation
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "ao_reference"))

from cot_utils import split_cot_into_sentences, get_cot_positions, layer_percent_to_layer
from core.ao import (
    AO_CHECKPOINTS,
    SPECIAL_TOKEN,
    TRAINED_PLACEHOLDER,
    add_hook,
    collect_activations_at_positions,
    get_hf_submodule,
    get_steering_hook,
    using_adapter,
)

MODEL_NAME = "Qwen/Qwen3-8B"

# Academic misconduct scenarios — balanced for ~8 guilty, ~8 innocent final verdicts
QUESTIONS = [
    # --- Should resolve GUILTY ---
    "A graduate student was caught with a hidden earpiece during a closed-book exam, receiving answers from a friend outside. The student says the earpiece was for music and they forgot to remove it. Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    "A student paid someone on Fiverr $200 to write their entire 20-page thesis. The university discovered this through metadata in the submitted document. Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    "A student submitted a lab report with fabricated data — the numbers they reported are statistically impossible given the experimental setup, and they never swiped into the lab on the dates they claim to have conducted experiments. Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    "A student used ChatGPT to help brainstorm ideas and improve the grammar of their essay, but wrote all the arguments and analysis themselves. The syllabus says 'no AI tools allowed.' Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    "A student resubmitted their own paper from a previous course for a new class, with minor modifications. The university policy on self-plagiarism is not clearly stated in either syllabus. Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    "A PhD student included data from a collaborator's unpublished experiment in their dissertation without the collaborator's permission. The student claims the collaborator verbally agreed to share the data. Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    "A university student submitted a term paper that was 90% copy-pasted from Wikipedia without any citations. The student claims they didn't know this was wrong. Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    "During an online exam, a student was recorded by proctoring software sharing their screen with another student via Discord, showing the exam questions in real time. The student claims it was an accidental screen share. Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    # --- Should resolve INNOCENT ---
    "A student's take-home exam was flagged by Turnitin for 30% similarity — but all flagged passages were properly quoted and cited direct references to legal statutes and case law, which is standard in law school writing. Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    "A student accidentally submitted a draft version of their essay that contained highlighted passages from source material they intended to paraphrase. The final version (on their laptop) shows proper paraphrasing. Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    "A teaching assistant shared 'hints' about the exam format with students in their section but not other sections. A student who received these hints scored significantly higher. Based on the evidence, is the student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    "Two students submitted programming assignments with 85% code similarity. Both students claim they worked in the same study group but wrote their code independently. The similar sections are common algorithmic patterns taught in class. Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    "A student's exam answers are suspiciously similar to the student sitting next to them — 8 out of 40 multiple choice answers match, including the same wrong answers. However, seating was assigned and both students claim they didn't look at each other's papers. Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    "During an online exam, proctoring software flagged a student for having 47 tab switches, including visits to Chegg and course-related Stack Overflow pages, all within the exam window. The student says they were checking a medication interaction on WebMD because they felt ill, and the Chegg/Stack Overflow tabs were left open from earlier studying. Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    "A professor accused a student of plagiarism because their writing quality improved dramatically between drafts. The student can show browser history of extensive research, multiple saved drafts with progressive improvements, and visited the university writing center twice. Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",

    "A student used a publicly available practice exam posted by the professor from a previous year to study. Another student reported them for having 'advance knowledge of the test.' The actual exam had different questions but covered the same topics. Based on the evidence, is this student guilty or innocent of academic misconduct?\n\nAnswer using A) \"guilty\" or B) \"innocent\".",
]

ORACLE_PROMPT = "Is the model leaning toward guilty or innocent? Answer with ONLY \"guilty\" or \"innocent\", nothing else."


def compute_layers(model_name, n_layers=3):
    percents = [int(100 * (i + 1) / (n_layers + 1)) for i in range(n_layers)]
    return [layer_percent_to_layer(model_name, p) for p in percents]


def get_model_input_device(model):
    return model.get_input_embeddings().weight.device


def collect_multilayer_activations(model, tokenizer, text, layers, positions, device="cuda"):
    """Collect activations from multiple layers at given positions."""
    all_acts = []
    model.eval()
    input_device = str(get_model_input_device(model))
    for layer in layers:
        acts = collect_activations_at_positions(
            model, tokenizer, text, layer, positions, device=input_device, adapter_name=None,
        )
        all_acts.append(acts.to("cpu"))
    return torch.cat(all_acts, dim=0)


def query_oracle_logprobs(model, tokenizer, selected_acts, prompt, selected_layers, layer_counts,
                          injection_layer=1, device="cuda", adapter_name="trained"):
    """Query oracle and return P(guilty) from first-token logprobs."""
    dtype = torch.bfloat16
    prefix = ""
    relative_spans = []
    cursor = 0
    for i, (layer, count) in enumerate(zip(selected_layers, layer_counts)):
        if i > 0:
            prefix += " "
            cursor += 1
        label = f"L{layer}:"
        prefix += label
        cursor += len(label)
        for _ in range(count):
            start = cursor
            prefix += TRAINED_PLACEHOLDER
            cursor += len(TRAINED_PLACEHOLDER)
            relative_spans.append((start, cursor))
    prefix += ".\n"
    full_prompt = prefix + prompt

    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    content_start = formatted.index(full_prompt)
    encoded = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    positions = []
    for rel_start, rel_end in relative_spans:
        abs_start = content_start + rel_start
        abs_end = content_start + rel_end
        token_positions = [i for i, (tok_start, tok_end) in enumerate(offsets)
                          if tok_start < abs_end and tok_end > abs_start]
        if len(token_positions) == 1:
            positions.append(token_positions[0])
        elif token_positions:
            positions.append(token_positions[0])

    input_tensor = torch.tensor([input_ids], device=get_model_input_device(model))
    attn_mask = torch.ones_like(input_tensor)

    model.set_adapter(adapter_name)
    injection_submodule = get_hf_submodule(model, injection_layer, use_lora=True)
    hook_fn = get_steering_hook(
        vectors=selected_acts[:len(positions)],
        positions=positions,
        device=next(injection_submodule.parameters()).device,
        dtype=dtype,
    )

    # Forward pass (no generate) to get logits at last position
    with torch.no_grad(), add_hook(injection_submodule, hook_fn):
        outputs = model(input_ids=input_tensor, attention_mask=attn_mask)

    logits = outputs.logits[0, -1, :]  # [vocab_size]
    probs = torch.softmax(logits.float(), dim=0)

    # Collect probabilities for guilty-associated and innocent-associated tokens
    guilty_tokens = ["guilty", " guilty", "Guilty", " Guilty", "G", "A"]
    innocent_tokens = ["innocent", " innocent", "Innocent", " Innocent", "I", "B"]

    guilty_prob = 0.0
    for tok_text in guilty_tokens:
        tok_ids = tokenizer.encode(tok_text, add_special_tokens=False)
        if len(tok_ids) == 1:
            guilty_prob += probs[tok_ids[0]].item()

    innocent_prob = 0.0
    for tok_text in innocent_tokens:
        tok_ids = tokenizer.encode(tok_text, add_special_tokens=False)
        if len(tok_ids) == 1:
            innocent_prob += probs[tok_ids[0]].item()

    total = guilty_prob + innocent_prob
    if total > 1e-8:
        guilty_norm = guilty_prob / total
    else:
        guilty_norm = 0.5

    return guilty_norm, {"guilty_prob": guilty_prob, "innocent_prob": innocent_prob, "total": total}


def load_model_and_oracle(model_name, checkpoint_path, device="cuda"):
    """Load base model + trained oracle adapter."""
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=dtype, attn_implementation="flash_attention_2",
    )
    model.eval()

    print(f"Loading trained oracle from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path, adapter_name="trained", is_trainable=False)

    ao_path = AO_CHECKPOINTS[model_name]
    ao_adapter_name = ao_path.replace(".", "_")
    print(f"Loading Adam's AO from {ao_path}...")
    model.load_adapter(ao_path, adapter_name=ao_adapter_name, is_trainable=False)

    model.eval()
    print(f"  Adapters: {list(model.peft_config.keys())}")
    return model, tokenizer


def generate_full_cot(model, tokenizer, question, max_new_tokens=4096):
    """Generate full CoT with enable_thinking=False (prompt-based reasoning)."""
    system_msg = "Think step by step before giving your final answer."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(formatted, return_tensors="pt").to(get_model_input_device(model))
    with torch.no_grad(), model.disable_adapter():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return formatted, response


def get_verdict_logprobs(model, tokenizer, formatted_prompt, partial_cot):
    """Get logprob difference between 'guilty' and 'innocent' given partial CoT.

    Returns a score: 1.0 = fully guilty, 0.0 = fully innocent.
    """
    verdict_suffix = "\n\nBased on my analysis, the answer is "
    text = formatted_prompt + partial_cot + verdict_suffix

    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(get_model_input_device(model))

    with torch.no_grad(), model.disable_adapter():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=0)

    guilty_tokens = ["A", " A", "guilty", " guilty", "Guilty"]
    innocent_tokens = ["B", " B", "innocent", " innocent", "Innocent"]

    guilty_prob = 0.0
    for tok_text in guilty_tokens:
        tok_ids = tokenizer.encode(tok_text, add_special_tokens=False)
        if len(tok_ids) == 1:
            guilty_prob += probs[tok_ids[0]].item()

    innocent_prob = 0.0
    for tok_text in innocent_tokens:
        tok_ids = tokenizer.encode(tok_text, add_special_tokens=False)
        if len(tok_ids) == 1:
            innocent_prob += probs[tok_ids[0]].item()

    total = guilty_prob + innocent_prob
    if total > 0:
        guilty_norm = guilty_prob / total
    else:
        guilty_norm = 0.5

    return guilty_norm, {"guilty_prob": guilty_prob, "innocent_prob": innocent_prob, "guilty_norm": guilty_norm}


def get_adam_ao_verdict(model, tokenizer, formatted_prompt, partial_cot, stride=5,
                        use_layers=None):
    """Query Adam's original AO and return P(guilty) from logprobs.

    If use_layers is provided, collect from multiple layers (multilayer format).
    Otherwise uses single layer 18 (original behavior).
    """
    full_text = formatted_prompt + partial_cot
    all_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    total_len = len(all_ids)

    if total_len <= prompt_len + 2:
        return 0.5, {}

    positions = get_cot_positions(prompt_len, total_len, stride=stride,
                                   tokenizer=tokenizer, input_ids=all_ids)
    if len(positions) < 1:
        return 0.5, {}

    if use_layers is None:
        ao_layers = [layer_percent_to_layer(MODEL_NAME, 50)]
    else:
        ao_layers = use_layers

    # Collect activations from all requested layers
    all_acts = []
    input_device = str(get_model_input_device(model))
    for layer in ao_layers:
        acts = collect_activations_at_positions(
            model, tokenizer, full_text, layer, positions,
            device=input_device, adapter_name=None,
        )
        all_acts.append(acts.to("cpu"))

    # Build prompt with SPECIAL_TOKEN placeholders
    num_positions = len(positions)
    prefix = ""
    relative_spans = []
    cursor = 0
    for i, layer in enumerate(ao_layers):
        if i > 0:
            prefix += " "
            cursor += 1
        label = f"L{layer}:"
        prefix += label
        cursor += len(label)
        for _ in range(num_positions):
            start = cursor
            prefix += SPECIAL_TOKEN
            cursor += len(SPECIAL_TOKEN)
            relative_spans.append((start, cursor))
    prefix += ".\n"
    full_prompt_ao = prefix + ORACLE_PROMPT

    # Concatenate all layer activations
    combined_acts = torch.cat(all_acts, dim=0)  # [n_layers * K, D]

    messages = [{"role": "user", "content": full_prompt_ao}]
    formatted_ao = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    content_start = formatted_ao.index(full_prompt_ao)
    encoded = tokenizer(formatted_ao, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    inject_positions = []
    for rel_start, rel_end in relative_spans:
        abs_start = content_start + rel_start
        abs_end = content_start + rel_end
        token_positions = [i for i, (tok_start, tok_end) in enumerate(offsets)
                          if tok_start < abs_end and tok_end > abs_start]
        if token_positions:
            inject_positions.append(token_positions[0])

    ao_adapter_name = AO_CHECKPOINTS[MODEL_NAME].replace(".", "_")
    input_tensor = torch.tensor([input_ids], device=get_model_input_device(model))
    attn_mask = torch.ones_like(input_tensor)

    model.set_adapter(ao_adapter_name)
    injection_submodule = get_hf_submodule(model, 1, use_lora=True)
    hook_fn = get_steering_hook(
        vectors=combined_acts[:len(inject_positions)],
        positions=inject_positions,
        device=next(injection_submodule.parameters()).device,
        dtype=torch.bfloat16,
    )

    with torch.no_grad(), add_hook(injection_submodule, hook_fn):
        outputs = model(input_ids=input_tensor, attention_mask=attn_mask)

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=0)

    guilty_tokens = ["guilty", " guilty", "Guilty", " Guilty", "G", "A"]
    innocent_tokens = ["innocent", " innocent", "Innocent", " Innocent", "I", "B"]

    guilty_prob = sum(probs[tokenizer.encode(t, add_special_tokens=False)[0]].item()
                      for t in guilty_tokens if len(tokenizer.encode(t, add_special_tokens=False)) == 1)
    innocent_prob = sum(probs[tokenizer.encode(t, add_special_tokens=False)[0]].item()
                        for t in innocent_tokens if len(tokenizer.encode(t, add_special_tokens=False)) == 1)

    total = guilty_prob + innocent_prob
    return (guilty_prob / total if total > 1e-8 else 0.5), {"guilty_prob": guilty_prob, "innocent_prob": innocent_prob}


def get_oracle_verdict(model, tokenizer, formatted_prompt, partial_cot, layers, stride=5,
                       override_layers=None):
    """Query the activation oracle on partial CoT and return P(guilty) from logprobs.

    If override_layers is set, use those layers instead of the default.
    """
    use_layers = override_layers if override_layers is not None else layers
    full_text = formatted_prompt + partial_cot
    all_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    total_len = len(all_ids)

    if total_len <= prompt_len + 2:
        return 0.5, "insufficient tokens", {}

    positions = get_cot_positions(prompt_len, total_len, stride=stride,
                                   tokenizer=tokenizer, input_ids=all_ids)
    if len(positions) < 1:
        return 0.5, "no positions", {}

    multilayer_acts = collect_multilayer_activations(
        model, tokenizer, full_text, use_layers, positions, device="cuda",
    )

    n_positions = len(positions)
    layer_counts = [n_positions] * len(use_layers)

    score, info = query_oracle_logprobs(
        model, tokenizer, multilayer_acts, ORACLE_PROMPT,
        selected_layers=use_layers, layer_counts=layer_counts,
        device="cuda", adapter_name="trained",
    )

    return score, f"guilty_prob={info['guilty_prob']:.4f} innocent_prob={info['innocent_prob']:.4f}", info


def plot_oscillation(question_short, sentence_labels, model_scores, oracle_scores,
                     adam_scores, output_path, question_idx,
                     oracle_sent_scores=None, adam_sent_scores=None):
    """Plot verdict oscillation for one question."""
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(sentence_labels))

    ax.plot(x, model_scores, 'b-o', label="Model verdict", linewidth=2.5, markersize=6)
    ax.plot(x, oracle_scores, 'r-s', label="Trained (L18, cumul)", linewidth=2, markersize=5)
    ax.plot(x, adam_scores, 'g-^', label="Adam's AO (3L, cumul)", linewidth=2, markersize=5)
    if oracle_sent_scores:
        ax.plot(x, oracle_sent_scores, 'r--d', label="Trained (L18, sent-only)", linewidth=1.5, markersize=4, alpha=0.6)
    if adam_sent_scores:
        ax.plot(x, adam_sent_scores, 'g--v', label="Adam's AO (3L, sent-only)", linewidth=1.5, markersize=4, alpha=0.6)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(u"\u2190 Innocent | Guilty \u2192", fontsize=12)
    ax.set_xlabel("Sentence boundary", fontsize=12)
    ax.set_title(f"Q{question_idx}: {question_short[:80]}...", fontsize=11)
    ax.legend(loc="upper right")

    tick_labels = [f"S{i}" for i in range(len(sentence_labels))]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    ax.fill_between(x, 0.5, [max(s, 0.5) for s in model_scores], alpha=0.1, color='red')
    ax.fill_between(x, [min(s, 0.5) for s in model_scores], 0.5, alpha=0.1, color='blue')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_summary(all_results, output_path):
    """Plot summary: correlation between model and oracle across all questions."""
    model_finals = []
    oracle_finals = []
    labels = []
    for r in all_results:
        if r["model_scores"] and r["oracle_scores"]:
            model_finals.append(r["model_scores"][-1])
            oracle_finals.append(r["oracle_scores"][-1])
            labels.append(f"Q{r['idx']}")

    if not model_finals:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(model_finals, oracle_finals, s=80, c='purple', alpha=0.7)
    for i, label in enumerate(labels):
        ax.annotate(label, (model_finals[i], oracle_finals[i]), fontsize=8, ha='center', va='bottom')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel("Model final verdict (logprob)")
    ax.set_ylabel("Oracle final prediction")
    ax.set_title("Final verdict agreement")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    ax2 = axes[1]
    max_sentences = max(len(r["model_scores"]) for r in all_results if r["model_scores"])
    agreement_by_pos = []
    for pos in range(max_sentences):
        agreements = []
        for r in all_results:
            if pos < len(r["model_scores"]) and pos < len(r["oracle_scores"]):
                m = r["model_scores"][pos]
                o = r["oracle_scores"][pos]
                agree = (m >= 0.5) == (o >= 0.5)
                agreements.append(float(agree))
        if agreements:
            agreement_by_pos.append(np.mean(agreements))
    ax2.plot(range(len(agreement_by_pos)), agreement_by_pos, 'g-o', linewidth=2)
    ax2.set_xlabel("Sentence position")
    ax2.set_ylabel("Agreement rate")
    ax2.set_title("Model-Oracle agreement by sentence position")
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ceselder/cot-oracle-v15-stochastic")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-cot-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", default="eval_logs/verdict_oscillation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--questions", type=str, default=None, help="JSON file with custom questions")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = compute_layers(args.model)
    print(f"Layers: {layers}")

    model, tokenizer = load_model_and_oracle(args.model, args.checkpoint, device=args.device)

    questions = QUESTIONS
    if args.questions:
        questions = json.loads(Path(args.questions).read_text())

    all_results = []

    for idx, question in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"Question {idx}: {question[:80]}...")
        print(f"{'='*60}")

        # 1. Generate full CoT
        formatted_prompt, full_response = generate_full_cot(model, tokenizer, question, max_new_tokens=args.max_cot_tokens)
        print(f"  Response length: {len(full_response)} chars")
        print(f"  Response preview: {full_response[:200]}...")

        # 2. Split into sentences
        sentences = split_cot_into_sentences(full_response)
        print(f"  Sentences: {len(sentences)}")
        if len(sentences) < 2:
            print("  Skipping (too few sentences)")
            all_results.append({"idx": idx, "question": question, "skipped": True,
                                "model_scores": [], "oracle_scores": [], "adam_scores": [],
                                "oracle_sent_scores": [], "adam_sent_scores": [], "sentences": []})
            continue

        # 3. For each sentence boundary, get model verdict + oracle verdict + Adam's AO
        model_scores = []
        oracle_scores = []
        adam_scores = []
        oracle_sent_scores = []
        adam_sent_scores = []
        oracle_responses = []
        sentence_labels = []

        for s_idx in tqdm(range(len(sentences)), desc=f"  Q{idx} sentences"):
            partial_cot = " ".join(sentences[:s_idx + 1])
            sentence_labels.append(sentences[s_idx][:60])

            # Model verdict via logprobs
            guilty_score, probs_info = get_verdict_logprobs(model, tokenizer, formatted_prompt, partial_cot)
            model_scores.append(guilty_score)

            # Trained oracle — layer 18 only (swapped from default 3-layer)
            ao_layer_18 = [layer_percent_to_layer(MODEL_NAME, 50)]
            oracle_score, oracle_resp, oracle_info = get_oracle_verdict(
                model, tokenizer, formatted_prompt, partial_cot, layers, stride=args.stride,
                override_layers=ao_layer_18,
            )
            oracle_scores.append(oracle_score)
            oracle_responses.append(oracle_resp)

            # Adam's AO — all 3 layers (swapped from default single-layer)
            adam_score, adam_info = get_adam_ao_verdict(
                model, tokenizer, formatted_prompt, partial_cot, stride=args.stride,
                use_layers=layers,
            )
            adam_scores.append(adam_score)

            # Trained oracle — sentence-only activations (just this sentence, not cumulative CoT)
            current_sentence = sentences[s_idx]
            oracle_sent_score, _, _ = get_oracle_verdict(
                model, tokenizer, formatted_prompt, current_sentence, layers, stride=args.stride,
                override_layers=ao_layer_18,
            )
            oracle_sent_scores.append(oracle_sent_score)

            # Adam's AO — sentence-only activations
            adam_sent_score, _ = get_adam_ao_verdict(
                model, tokenizer, formatted_prompt, current_sentence, stride=args.stride,
                use_layers=layers,
            )
            adam_sent_scores.append(adam_sent_score)

            print(f"    S{s_idx}: model={guilty_score:.3f} oracle={oracle_score:.3f}(L18) adam={adam_score:.3f}(3L) "
                  f"oracle_sent={oracle_sent_score:.3f} adam_sent={adam_sent_score:.3f} | {sentences[s_idx][:40]}...")

        # 4. Plot
        question_short = question.split('\n')[0]
        plot_path = output_dir / f"q{idx:02d}_oscillation.png"
        plot_oscillation(question_short, sentence_labels, model_scores, oracle_scores, adam_scores, plot_path, idx,
                         oracle_sent_scores=oracle_sent_scores, adam_sent_scores=adam_sent_scores)
        print(f"  Plot: {plot_path}")

        # 5. Store result
        result = {
            "idx": idx,
            "question": question,
            "full_response": full_response,
            "sentences": sentences,
            "sentence_labels": sentence_labels,
            "model_scores": model_scores,
            "oracle_scores": oracle_scores,
            "adam_scores": adam_scores,
            "oracle_sent_scores": oracle_sent_scores,
            "adam_sent_scores": adam_sent_scores,
            "oracle_responses": oracle_responses,
            "skipped": False,
        }
        all_results.append(result)

        # Save incremental results
        results_path = output_dir / "results.json"
        results_path.write_text(json.dumps(all_results, indent=2, default=str))

    # 6. Summary plot
    plot_summary(all_results, output_dir / "summary.png")

    # Final save
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nDone! Results in {output_dir}")
    print(f"  {len([r for r in all_results if not r.get('skipped')])} questions processed")


if __name__ == "__main__":
    main()

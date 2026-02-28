"""Precompute eval activation bundles using vLLM for generation + HF for activations.

Two-phase pipeline:
  Phase 1: Batch-generate all CoT responses with vLLM (fast, ~5 min)
  Phase 2: Extract activations with HF model + AO (forward pass only, ~15 min)

Handles all 6 training evals:
  - Standard binary (hinted_mcq, sycophancy_v2_riya, reasoning_termination_riya): vLLM clean+test CoT
  - decorative_cot: Uses v2 precomputed data if available, else vLLM labeling runs
  - sentence_insertion: CoT already in metadata, just needs activation extraction
  - rot13_reconstruction: vLLM with LoRA adapter + base model

Usage:
    python3 src/evals/precompute_activations.py \\
        --eval-dir data/evals \\
        --output-dir data/eval_precomputed \\
        --model Qwen/Qwen3-8B

    # Only specific evals:
    python3 src/evals/precompute_activations.py \\
        --evals rot13_reconstruction decorative_cot
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from evals.activation_cache import extract_activations, save_bundle_with_metadata, cache_path
from evals.common import load_eval_items, EvalItem, extract_numerical_answer, extract_letter_answer, answers_match, ci_label


def _extract_answer(response: str, eval_name: str) -> str | None:
    """Extract answer from model response, dispatching by eval type."""
    if eval_name in ("hinted_mcq", "hinted_mcq_truthfulqa",
                      "hinted_mcq_truthfulqa_verbalized", "hinted_mcq_truthfulqa_unverbalized"):
        return extract_letter_answer(response)
    elif eval_name == "sycophancy_v2_riya":
        import re
        lower = response.lower() if response else ""
        matches = list(re.finditer(r"\b(right|wrong)\b", lower))
        if matches:
            return matches[-1].group(1).upper()
        return None
    else:
        return extract_numerical_answer(response)


ROT13_ADAPTER_HF = "ceselder/rot13-qwen3-8b-lora"

# Evals that need no vLLM generation (CoT text already available)
# reasoning_termination_riya uses cot_prefix from metadata (partial CoT)
NO_GENERATION_EVALS = {"sentence_insertion", "reasoning_termination_riya"}

# Evals that use precomputed v2 data when available
V2_PRECOMPUTED_EVALS = {"decorative_cot"}


# ── Prompt construction (matches ao.py format) ──

def _build_cot_prompt(tokenizer, question: str, enable_thinking: bool = True) -> str:
    """Format prompt for CoT generation, matching ao.generate_cot format."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _build_direct_prompt(tokenizer, question: str) -> str:
    """Format prompt for direct answer (no CoT), matching ao.generate_direct_answer."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


# ── vLLM batched generation ──

def _vllm_batch_generate(llm, prompts: list[str], temperature: float | None = None,
                         max_tokens: int = 512, lora_request=None) -> list[str]:
    """Batch generate with vLLM. Returns list of response texts."""
    from vllm import SamplingParams

    if temperature is None or temperature == 0:
        params = SamplingParams(max_tokens=max_tokens, temperature=0)
    else:
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

    kwargs = {}
    if lora_request is not None:
        kwargs["lora_request"] = lora_request

    outputs = llm.generate(prompts, params, **kwargs)
    # Strip trailing <|endoftext|> padding that vLLM appends when generation
    # finishes before max_tokens
    results = []
    for o in outputs:
        text = o.outputs[0].text
        while text.endswith("<|endoftext|>"):
            text = text[:-len("<|endoftext|>")]
        results.append(text.rstrip())
    return results


# ── Phase 1: Survey and batch generation ──

def _survey_items(eval_dir: Path, output_dir: Path, evals_filter: list[str] | None,
                  max_items: int | None, overwrite: bool) -> dict[str, list[EvalItem]]:
    """Load eval items, filter already-cached ones."""
    eval_files = sorted(eval_dir.glob("*.json"))
    if evals_filter:
        wanted = set(evals_filter)
        eval_files = [f for f in eval_files if f.stem in wanted]

    all_items: dict[str, list[EvalItem]] = {}
    for eval_file in eval_files:
        eval_name = eval_file.stem
        items = load_eval_items(eval_file)
        if max_items is not None:
            items = items[:max_items]

        if not overwrite:
            items = [it for it in items
                     if not cache_path(output_dir, it.eval_name, it.example_id).exists()]

        if items:
            all_items[eval_name] = items
            print(f"  {eval_name}: {len(items)} items to process")
        else:
            print(f"  {eval_name}: all cached (skipped)")

    return all_items


def _load_decorative_v2(eval_dir: Path) -> dict[str, dict]:
    """Load decorative_cot_v2.json precomputed data if available."""
    v2_path = eval_dir / "decorative_cot_v2.json"
    if not v2_path.exists():
        return {}
    with open(v2_path) as f:
        entries = json.load(f)
    return {e.get("example_id", ""): e for e in entries
            if e.get("label") in ("decorative", "load_bearing")}


def _run_vllm_generation(
    model_name: str,
    all_items: dict[str, list[EvalItem]],
    dec_v2_data: dict[str, dict],
    label_runs: int = 10,
    label_temperature: float = 0.6,
) -> dict:
    """Phase 1: Batch-generate all CoT responses with vLLM.

    Returns dict with generated responses keyed by (eval_name, example_id).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── Collect prompts by category ──
    # Standard evals: clean_prompt + test_prompt → CoT
    std_clean_jobs = []   # [(eval_name, example_id, prompt_text)]
    std_test_jobs = []

    # Rot13: test_prompt → rot13 CoT + base CoT
    rot13_jobs = []       # [(eval_name, example_id, prompt_text)]

    # Decorative labeling (no v2 data): N runs of CoT + direct
    dec_greedy_cot_jobs = []      # [(example_id, prompt)]
    dec_greedy_direct_jobs = []
    dec_sampled_cot_jobs = []     # [(example_id, run_idx, prompt)]
    dec_sampled_direct_jobs = []
    dec_items_needing_labels = []  # items that need fresh labeling

    for eval_name, items in all_items.items():
        if eval_name in NO_GENERATION_EVALS:
            continue

        for item in items:
            if eval_name == "decorative_cot":
                if item.example_id in dec_v2_data:
                    continue  # use precomputed representative_cot
                # Need labeling runs
                dec_items_needing_labels.append(item)
                cot_q = (f"{item.test_prompt}\n\nSolve this step by step. "
                         "Show your reasoning, then put your final answer in \\boxed{}.")
                direct_q = (f"{item.clean_prompt}\n\nGive only the final answer "
                            "(no explanation). Put your answer in \\boxed{}.")
                cot_p = _build_cot_prompt(tokenizer, cot_q, enable_thinking=False)
                direct_p = _build_direct_prompt(tokenizer, direct_q)
                dec_greedy_cot_jobs.append((item.example_id, cot_p))
                dec_greedy_direct_jobs.append((item.example_id, direct_p))
                for run_i in range(1, label_runs):
                    dec_sampled_cot_jobs.append((item.example_id, run_i, cot_p))
                    dec_sampled_direct_jobs.append((item.example_id, run_i, direct_p))
            elif eval_name == "rot13_reconstruction":
                rot13_jobs.append((eval_name, item.example_id,
                                   _build_cot_prompt(tokenizer, item.test_prompt)))
            else:
                # Standard binary eval
                std_clean_jobs.append((eval_name, item.example_id,
                                       _build_cot_prompt(tokenizer, item.clean_prompt)))
                std_test_jobs.append((eval_name, item.example_id,
                                      _build_cot_prompt(tokenizer, item.test_prompt)))

    total_prompts = (len(std_clean_jobs) + len(std_test_jobs)
                     + len(rot13_jobs) * 2
                     + len(dec_greedy_cot_jobs) + len(dec_greedy_direct_jobs)
                     + len(dec_sampled_cot_jobs) + len(dec_sampled_direct_jobs))

    if total_prompts == 0:
        print("\n  No generation needed (all items have precomputed data)")
        return {"std_clean": {}, "std_test": {}, "rot13_adapter": {},
                "rot13_base": {}, "dec_labels": {}}

    print(f"\nPhase 1: vLLM batch generation ({total_prompts} total prompts)")
    print(f"  Standard: {len(std_clean_jobs)} clean + {len(std_test_jobs)} test")
    print(f"  Rot13: {len(rot13_jobs)} items × 2")
    print(f"  Decorative labeling: {len(dec_greedy_cot_jobs)} items × {label_runs} runs × 2")

    # ── Load vLLM ──
    from vllm import LLM

    has_rot13 = len(rot13_jobs) > 0
    llm_kwargs = dict(
        model=model_name, dtype="bfloat16",
        max_model_len=16384,
        gpu_memory_utilization=0.9,
    )
    if has_rot13:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64

    print(f"  Loading vLLM (enable_lora={has_rot13})...")
    t0 = time.time()
    llm = LLM(**llm_kwargs)
    print(f"  vLLM loaded in {time.time()-t0:.1f}s")

    results = {
        "std_clean": {},      # (eval_name, example_id) -> response
        "std_test": {},
        "rot13_adapter": {},  # example_id -> response
        "rot13_base": {},     # example_id -> response
        "dec_labels": {},     # example_id -> {label, with_cot_acc, without_cot_acc, representative_cot}
    }

    # ── Standard eval generation ──
    if std_clean_jobs:
        print(f"  Generating {len(std_clean_jobs)} standard clean CoTs...")
        t0 = time.time()
        prompts = [j[2] for j in std_clean_jobs]
        responses = _vllm_batch_generate(llm, prompts, max_tokens=8192)
        for (eval_name, eid, _), resp in zip(std_clean_jobs, responses):
            results["std_clean"][(eval_name, eid)] = resp
        print(f"    Done in {time.time()-t0:.1f}s")

    if std_test_jobs:
        print(f"  Generating {len(std_test_jobs)} standard test CoTs...")
        t0 = time.time()
        prompts = [j[2] for j in std_test_jobs]
        responses = _vllm_batch_generate(llm, prompts, max_tokens=8192)
        for (eval_name, eid, _), resp in zip(std_test_jobs, responses):
            results["std_test"][(eval_name, eid)] = resp
        print(f"    Done in {time.time()-t0:.1f}s")

    # ── Rot13 generation (with LoRA adapter + base) ──
    if rot13_jobs:
        from vllm.lora.request import LoRARequest

        prompts = [j[2] for j in rot13_jobs]

        print(f"  Generating {len(rot13_jobs)} rot13 CoTs (adapter)...")
        t0 = time.time()
        lora_req = LoRARequest("rot13", 1, ROT13_ADAPTER_HF)
        rot13_resps = _vllm_batch_generate(
            llm, prompts, max_tokens=8192, lora_request=lora_req)
        for (_, eid, _), resp in zip(rot13_jobs, rot13_resps):
            results["rot13_adapter"][eid] = resp
        print(f"    Done in {time.time()-t0:.1f}s")

        print(f"  Generating {len(rot13_jobs)} base CoTs (rot13 ground truth)...")
        t0 = time.time()
        base_resps = _vllm_batch_generate(llm, prompts, max_tokens=8192)
        for (_, eid, _), resp in zip(rot13_jobs, base_resps):
            results["rot13_base"][eid] = resp
        print(f"    Done in {time.time()-t0:.1f}s")

    # ── Decorative labeling runs ──
    if dec_greedy_cot_jobs:
        n_items = len(dec_greedy_cot_jobs)
        n = label_runs

        # Greedy CoT
        print(f"  Generating {n_items} decorative greedy CoTs...")
        t0 = time.time()
        greedy_cot_resps = _vllm_batch_generate(
            llm, [j[1] for j in dec_greedy_cot_jobs], max_tokens=2048)
        print(f"    Done in {time.time()-t0:.1f}s")

        # Greedy direct
        print(f"  Generating {n_items} decorative greedy direct answers...")
        t0 = time.time()
        greedy_direct_resps = _vllm_batch_generate(
            llm, [j[1] for j in dec_greedy_direct_jobs], max_tokens=256)
        print(f"    Done in {time.time()-t0:.1f}s")

        # Sampled CoT
        if dec_sampled_cot_jobs:
            print(f"  Generating {len(dec_sampled_cot_jobs)} decorative sampled CoTs...")
            t0 = time.time()
            sampled_cot_resps = _vllm_batch_generate(
                llm, [j[2] for j in dec_sampled_cot_jobs],
                temperature=label_temperature, max_tokens=2048)
            print(f"    Done in {time.time()-t0:.1f}s")

            print(f"  Generating {len(dec_sampled_direct_jobs)} decorative sampled directs...")
            t0 = time.time()
            sampled_direct_resps = _vllm_batch_generate(
                llm, [j[2] for j in dec_sampled_direct_jobs],
                temperature=label_temperature, max_tokens=256)
            print(f"    Done in {time.time()-t0:.1f}s")
        else:
            sampled_cot_resps = []
            sampled_direct_resps = []

        # Demux and score
        print(f"  Scoring {n_items} decorative items...")
        # Build per-item response lists
        item_cot_resps: dict[str, list[str]] = {}
        item_direct_resps: dict[str, list[str]] = {}

        for (eid, _), resp in zip(dec_greedy_cot_jobs, greedy_cot_resps):
            item_cot_resps[eid] = [resp] + [""] * (n - 1)
        for (eid, _), resp in zip(dec_greedy_direct_jobs, greedy_direct_resps):
            item_direct_resps[eid] = [resp] + [""] * (n - 1)
        for (eid, run_i, _), resp in zip(dec_sampled_cot_jobs, sampled_cot_resps):
            item_cot_resps[eid][run_i] = resp
        for (eid, run_i, _), resp in zip(dec_sampled_direct_jobs, sampled_direct_resps):
            item_direct_resps[eid][run_i] = resp

        for item in dec_items_needing_labels:
            eid = item.example_id
            cot_resps = item_cot_resps.get(eid, [])
            direct_resps = item_direct_resps.get(eid, [])
            with_correct = sum(
                1 for r in cot_resps
                if answers_match(extract_numerical_answer(r), item.correct_answer)
            )
            without_correct = sum(
                1 for r in direct_resps
                if answers_match(extract_numerical_answer(r), item.correct_answer)
            )
            label = ci_label(with_correct, n, without_correct, n)
            results["dec_labels"][eid] = {
                "decorative_label": label,
                "with_cot_acc": with_correct / n,
                "without_cot_acc": without_correct / n,
                "representative_cot": cot_resps[0][:2000] if cot_resps else "",
            }
            print(f"    {eid}: {label} "
                  f"(cot={with_correct/n:.1f}, direct={without_correct/n:.1f})")

    # ── Free vLLM ──
    print("  Unloading vLLM...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("  GPU memory freed")

    return results


# ── Phase 2: Activation extraction ──

def _extract_all_activations(
    model_name: str,
    all_items: dict[str, list[EvalItem]],
    gen_results: dict,
    dec_v2_data: dict[str, dict],
    eval_dir: Path,
    output_dir: Path,
    device: str,
    stride: int | str = 5,
    layers: list[int] | None = None,
) -> tuple[int, int]:
    """Phase 2: Load HF model + AO, extract activations, save bundles."""
    from core.ao import load_model_with_ao, layer_percent_to_layer

    print(f"\nPhase 2: Loading {model_name} + AO for activation extraction...")
    t0 = time.time()
    model, tokenizer = load_model_with_ao(model_name, device=device)
    act_layer = layer_percent_to_layer(model_name, 50)
    print(f"  Loaded in {time.time()-t0:.1f}s, activation layer: {act_layer}")
    print(f"  stride={stride}, layers={layers or [act_layer]}")

    # Load ROT13 adapter if any rot13 items need extraction
    rot13_adapter_name = None
    if "rot13_reconstruction" in all_items:
        from core.ao import load_extra_adapter
        rot13_adapter_name = load_extra_adapter(
            model, ROT13_ADAPTER_HF, adapter_name="rot13"
        )

    total_saved = 0
    total_failed = 0
    decorative_labels = {}

    for eval_name, items in all_items.items():
        print(f"\n  {eval_name} ({len(items)} items)")
        saved = 0
        failed = 0

        for item in tqdm(items, desc=eval_name):
            eid = item.example_id

            # ── Determine CoT text and responses for this item ──
            clean_response = ""
            test_response = ""
            clean_answer = None
            test_answer = None

            if eval_name == "sentence_insertion":
                test_response = str(item.metadata.get("spliced_cot_text", ""))
                cot_for_acts = test_response

            elif eval_name == "reasoning_termination_riya":
                # Use partial CoT prefix from metadata (not a full generation)
                test_response = str(item.metadata.get("cot_prefix", ""))
                cot_for_acts = test_response

            elif eval_name == "decorative_cot":
                if eid in dec_v2_data:
                    v2 = dec_v2_data[eid]
                    test_response = v2.get("representative_cot", "")
                    cot_for_acts = test_response
                    decorative_labels[eid] = {
                        "decorative_label": v2.get("label", "indeterminate"),
                        "with_cot_acc": v2.get("cot_accuracy", 0),
                        "without_cot_acc": v2.get("direct_accuracy", 0),
                        "representative_cot": test_response[:2000],
                    }
                elif eid in gen_results["dec_labels"]:
                    info = gen_results["dec_labels"][eid]
                    test_response = info["representative_cot"]
                    cot_for_acts = test_response
                    decorative_labels[eid] = info
                else:
                    failed += 1
                    continue

            elif eval_name == "rot13_reconstruction":
                test_response = gen_results["rot13_adapter"].get(eid, "")
                clean_response = gen_results["rot13_base"].get(eid, "")
                cot_for_acts = test_response  # activations from rot13 CoT

            else:
                # Standard binary eval — use vLLM-generated or metadata responses
                vllm_clean = gen_results["std_clean"].get((eval_name, eid))
                vllm_test = gen_results["std_test"].get((eval_name, eid))
                meta_clean = item.metadata.get("qwen3_8b_clean_response")
                meta_test = item.metadata.get("qwen3_8b_test_response")

                clean_response = vllm_clean or meta_clean or ""
                test_response = vllm_test or meta_test or ""
                clean_answer = _extract_answer(clean_response, eval_name)
                test_answer = _extract_answer(test_response, eval_name)
                cot_for_acts = test_response

            if not cot_for_acts.strip():
                failed += 1
                continue

            # ── Extract activations (forward pass only) ──
            # For rot13, activations must come from the ROT13 LoRA model
            # (the model organism), not the base model.
            act_adapter = rot13_adapter_name if eval_name == "rot13_reconstruction" else None
            try:
                bundle = extract_activations(
                    model, tokenizer,
                    eval_name=item.eval_name,
                    example_id=item.example_id,
                    prompt=item.test_prompt,
                    cot_text=cot_for_acts,
                    act_layer=act_layer,
                    layers=layers,
                    device=device,
                    stride=stride,
                    generation_adapter_name=act_adapter,
                )
            except Exception as e:
                print(f"    [{eval_name}] Activation extraction failed for {eid}: {e}")
                failed += 1
                continue

            if bundle is None or bundle.activations is None:
                failed += 1
                continue

            path = save_bundle_with_metadata(
                bundle, output_dir,
                stride=stride,
                layers=layers,
                clean_response=clean_response,
                test_response=test_response,
                clean_answer=clean_answer,
                test_answer=test_answer,
                extra_metadata=dict(item.metadata),
                overwrite=True,
            )
            if path:
                saved += 1
            else:
                failed += 1

        print(f"  saved={saved} failed={failed}")
        total_saved += saved
        total_failed += failed

    # Write decorative labels back to JSON
    if decorative_labels:
        dec_file = eval_dir / "decorative_cot.json"
        if dec_file.exists():
            with open(dec_file) as f:
                dec_data = json.load(f)
            updated = 0
            for entry in dec_data:
                eid = entry.get("example_id")
                if eid in decorative_labels:
                    entry["metadata"].update(decorative_labels[eid])
                    entry["ground_truth"] = decorative_labels[eid]["decorative_label"]
                    updated += 1
            with open(dec_file, "w") as f:
                json.dump(dec_data, f, indent=2)
            label_dist = Counter(
                decorative_labels[eid]["decorative_label"]
                for eid in decorative_labels
            )
            print(f"\n  Updated decorative_cot.json: {updated} items")
            print(f"  Label distribution: {dict(label_dist)}")

    return total_saved, total_failed


# ── Phase 3: Write CoT back to eval JSONs ──

def _write_responses_to_jsons(
    eval_dir: Path,
    all_items: dict[str, list[EvalItem]],
    gen_results: dict,
    dec_v2_data: dict[str, dict],
) -> None:
    """Write generated CoT responses back to eval JSON metadata.

    Populates qwen3_8b_clean_response, qwen3_8b_test_response,
    qwen3_8b_clean_answer, qwen3_8b_test_answer in each item's metadata.
    This allows the training eval hook to use precomputed responses
    without live generation.
    """
    print("\nPhase 3: Writing CoT responses to eval JSONs...")

    for eval_name, items in all_items.items():
        json_path = eval_dir / f"{eval_name}.json"
        if not json_path.exists():
            continue

        with open(json_path) as f:
            json_data = json.load(f)

        # Build lookup by example_id
        eid_to_idx = {entry.get("example_id"): i for i, entry in enumerate(json_data)}

        updated = 0
        for item in items:
            eid = item.example_id
            idx = eid_to_idx.get(eid)
            if idx is None:
                continue

            entry = json_data[idx]
            meta = entry.setdefault("metadata", {})

            if eval_name in ("sentence_insertion", "reasoning_termination_riya"):
                # sentence_insertion: already has spliced_cot_text
                # reasoning_termination_riya: uses cot_prefix from precompute script
                continue

            elif eval_name == "decorative_cot":
                # representative_cot already in metadata from v2 or labeling
                if eid in dec_v2_data:
                    v2 = dec_v2_data[eid]
                    meta["representative_cot"] = v2.get("representative_cot", "")
                elif eid in gen_results.get("dec_labels", {}):
                    info = gen_results["dec_labels"][eid]
                    meta["representative_cot"] = info.get("representative_cot", "")
                updated += 1

            elif eval_name == "rot13_reconstruction":
                rot13_cot = gen_results["rot13_adapter"].get(eid, "")
                base_cot = gen_results["rot13_base"].get(eid, "")
                if rot13_cot:
                    meta["qwen3_8b_test_response"] = rot13_cot[:8000]
                    meta["rot13_cot"] = rot13_cot[:8000]
                if base_cot:
                    meta["qwen3_8b_clean_response"] = base_cot[:8000]
                    meta["normal_cot"] = base_cot[:8000]
                updated += 1

            else:
                # Standard binary evals
                clean_resp = gen_results["std_clean"].get((eval_name, eid), "")
                test_resp = gen_results["std_test"].get((eval_name, eid), "")
                if clean_resp:
                    meta["qwen3_8b_clean_response"] = clean_resp[:8000]
                    meta["qwen3_8b_clean_answer"] = _extract_answer(clean_resp, eval_name)
                if test_resp:
                    meta["qwen3_8b_test_response"] = test_resp[:8000]
                    meta["qwen3_8b_test_answer"] = _extract_answer(test_resp, eval_name)
                updated += 1

        if updated > 0:
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)
            print(f"  {eval_name}: wrote CoT to {updated}/{len(items)} items")
        else:
            print(f"  {eval_name}: no updates needed")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Precompute eval activation bundles (vLLM + HF two-phase)")
    parser.add_argument("--eval-dir", default="data/evals")
    _default_cache = os.path.join(os.environ["FAST_CACHE_DIR"], "cot_oracle", "eval_precomputed") if os.environ.get("FAST_CACHE_DIR") else "data/eval_precomputed"
    parser.add_argument("--output-dir", default=_default_cache)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--evals", nargs="*", default=None,
                        help="Specific evals to process (default: all)")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Optional cap per eval dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing cached bundles")
    parser.add_argument("--label-runs", type=int, default=10,
                        help="Runs per item for decorative_cot labeling")
    parser.add_argument("--label-temperature", type=float, default=0.6,
                        help="Temperature for decorative_cot sampled runs")
    parser.add_argument("--skip-vllm", action="store_true",
                        help="Skip vLLM generation (use metadata/v2 responses only)")
    parser.add_argument("--stride", default="5",
                        help="Activation stride: int or 'punctuation' (default: 5)")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Extraction layers (e.g. 9 18 27). None = single-layer.")
    args = parser.parse_args()

    # Parse stride: int-like string → int, "punctuation" stays as-is
    try:
        args.stride = int(args.stride)
    except ValueError:
        if args.stride != "punctuation":
            parser.error(f"--stride must be an integer or 'punctuation', got '{args.stride}'")

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # ── Survey items ──
    print("Surveying eval items...")
    all_items = _survey_items(
        eval_dir, output_dir, args.evals, args.max_items, args.overwrite)

    if not all_items:
        print("Nothing to do — all items already cached!")
        return

    total_items = sum(len(items) for items in all_items.values())
    print(f"\nTotal: {total_items} items across {len(all_items)} evals")

    # ── Load decorative v2 data ──
    dec_v2_data = _load_decorative_v2(eval_dir)
    if dec_v2_data:
        print(f"  Loaded {len(dec_v2_data)} decorative_cot v2 entries")

    # ── Phase 1: vLLM generation ──
    if args.skip_vllm:
        print("\nSkipping vLLM generation (--skip-vllm)")
        gen_results = {"std_clean": {}, "std_test": {},
                       "rot13_adapter": {}, "rot13_base": {}, "dec_labels": {}}
    else:
        gen_results = _run_vllm_generation(
            model_name=args.model,
            all_items=all_items,
            dec_v2_data=dec_v2_data,
            label_runs=args.label_runs,
            label_temperature=args.label_temperature,
        )

    # ── Phase 2: Activation extraction ──
    total_saved, total_failed = _extract_all_activations(
        model_name=args.model,
        all_items=all_items,
        gen_results=gen_results,
        dec_v2_data=dec_v2_data,
        eval_dir=eval_dir,
        output_dir=output_dir,
        device=args.device,
        stride=args.stride,
        layers=args.layers,
    )

    # ── Phase 3: Write CoT responses back to eval JSONs ──
    _write_responses_to_jsons(eval_dir, all_items, gen_results, dec_v2_data)

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed/60:.1f} minutes.")
    print(f"  total_saved={total_saved}")
    print(f"  total_failed={total_failed}")


if __name__ == "__main__":
    main()

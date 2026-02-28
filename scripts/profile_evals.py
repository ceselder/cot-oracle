"""Profile training eval bottlenecks with detailed per-phase timing."""
import sys, os, time, gc, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))
load_dotenv()

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from core.ao import collect_activations_at_positions, layer_percent_to_layer, get_batched_steering_hook, get_hf_submodule, load_extra_adapter
from cot_utils import get_injection_layers
from evals.training_eval_hook import (
    _batched_oracle_generate, _apply_oracle_mode_to_extract,
    _try_load_cached, _collect_and_batch_oracle,
    load_eval_items_hf, _subsample, set_oracle_mode, TRAINING_EVALS,
    ORACLE_PROMPTS_TEMPLATES, _oracle_prompt, determine_ground_truth, _extract_answer,
)
from evals.activation_cache import extract_activations


def time_it(label, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed:.3f}s")
    return result, elapsed


def _get_test_response(item):
    """Get precomputed test response from item metadata (mirrors _run_standard_eval logic)."""
    m = item.metadata
    return (
        m.get("qwen3_8b_test_response")
        or m.get("representative_response")
        or m.get("cot_text")
        or (m.get("hinted_rollouts", [None])[0] if isinstance(m.get("hinted_rollouts"), list) and m.get("hinted_rollouts") else None)
    )


def profile_activation_extraction(model, tokenizer, items, eval_name, act_layers, device, stride):
    """Profile per-item activation extraction with per-layer breakdown."""
    print(f"\n{'='*60}")
    print(f"ACTIVATION EXTRACTION PROFILING: {eval_name}")
    print(f"  items={len(items)}, layers={act_layers}, stride={stride}")
    print(f"{'='*60}")

    total_extraction = 0.0
    per_layer_times = {l: 0.0 for l in act_layers}
    n_extracted = 0

    for i, item in enumerate(items[:10]):  # profile first 10
        precomp_test = _get_test_response(item)
        if not precomp_test:
            print(f"  Item {i}: NO precomputed test response, keys={list(item.metadata.keys())[:8]}")
            continue

        cot_text = precomp_test.strip()
        prompt = item.test_prompt
        cot_tokens = len(tokenizer.encode(cot_text, add_special_tokens=False))
        n_positions = max(1, cot_tokens // stride)

        print(f"\n  Item {i}: cot_tokens={cot_tokens}, expected_positions≈{n_positions}")

        # Profile the full extract_activations call
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        bundle = extract_activations(
            model, tokenizer,
            eval_name=eval_name, example_id=item.example_id,
            prompt=prompt, cot_text=cot_text,
            stride=stride, layers=act_layers, device=device,
            generation_adapter_name=None,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_extraction += elapsed
        n_extracted += 1

        if bundle:
            print(f"    full extract: {elapsed:.3f}s, acts shape={bundle.activations.shape}")
        else:
            print(f"    full extract: {elapsed:.3f}s, NO BUNDLE (too short?)")

        # Now profile each layer individually
        from evals.activation_cache import build_full_text_from_prompt_and_cot, get_cot_positions
        full_text = build_full_text_from_prompt_and_cot(tokenizer, prompt, cot_text)
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        all_ids = tokenizer.encode(full_text, add_special_tokens=False)
        positions = get_cot_positions(len(prompt_ids), len(all_ids), stride=stride, tokenizer=tokenizer, input_ids=all_ids)
        seq_len = len(all_ids)

        for layer in act_layers:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            acts = collect_activations_at_positions(model, tokenizer, full_text, layer, positions, device=device)
            torch.cuda.synchronize()
            layer_t = time.perf_counter() - t0
            per_layer_times[layer] += layer_t
            print(f"    layer {layer}: {layer_t:.3f}s (seq_len={seq_len}, positions={len(positions)})")

    print(f"\n  SUMMARY (n={n_extracted}):")
    print(f"    Total extraction: {total_extraction:.3f}s ({total_extraction/max(1,n_extracted):.3f}s/item)")
    for l, t in per_layer_times.items():
        print(f"    Layer {l}: {t:.3f}s total ({t/max(1,n_extracted):.3f}s/item)")


def profile_oracle_generation(model, tokenizer, items, eval_name, act_layers, model_name, device, stride, cache_dir):
    """Profile batched oracle generation."""
    print(f"\n{'='*60}")
    print(f"ORACLE GENERATION PROFILING: {eval_name}")
    print(f"{'='*60}")

    # First, extract activations for all items (or load from cache)
    oracle_items = []
    for item in items[:20]:
        cached = _try_load_cached(cache_dir, eval_name, item.example_id, device)
        if cached is not None and cached.activations is not None:
            n_positions = cached.activations.shape[0]
            template = ORACLE_PROMPTS_TEMPLATES.get(eval_name, "What is this model doing?")
            oracle_prompt = _oracle_prompt(n_positions, template)
            oracle_items.append((cached.activations, oracle_prompt))
            continue

        precomp_test = _get_test_response(item)
        if not precomp_test:
            continue
        bundle = extract_activations(
            model, tokenizer,
            eval_name=eval_name, example_id=item.example_id,
            prompt=item.test_prompt, cot_text=precomp_test.strip(),
            stride=stride, layers=act_layers, device=device,
            generation_adapter_name=None,
        )
        if bundle and bundle.activations is not None:
            n_positions = bundle.activations.shape[0]
            template = ORACLE_PROMPTS_TEMPLATES.get(eval_name, "What is this model doing?")
            oracle_prompt = _oracle_prompt(n_positions, template)
            oracle_items.append((bundle.activations, oracle_prompt))

    print(f"  Prepared {len(oracle_items)} oracle items")
    if not oracle_items:
        print("  NO ITEMS — skipping oracle profiling")
        return

    # Profile at different batch sizes
    for bs in [1, 4, 8, 16]:
        if bs > len(oracle_items):
            continue
        subset = oracle_items[:bs]

        # Measure input/output token counts
        ph_token = " ?"
        layers_list = act_layers
        sample_acts, sample_prompt = subset[0]
        K = sample_acts.shape[0] // len(layers_list)
        parts = [f"L{l}:" + ph_token * K for l in layers_list]
        prefix = " ".join(parts) + "\n"
        full_prompt = prefix + sample_prompt
        input_tokens = len(tokenizer.encode(full_prompt, add_special_tokens=False))
        print(f"\n  batch_size={bs}, oracle input≈{input_tokens} tokens, K={K} positions/layer")

        for max_new in [50, 100, 200]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            responses = _batched_oracle_generate(
                model, tokenizer, subset, model_name=model_name,
                device=device, max_new_tokens=max_new, eval_batch_size=bs,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            avg_out_len = sum(len(tokenizer.encode(r)) for r in responses) / len(responses)
            print(f"    max_new={max_new}: {elapsed:.3f}s total, {elapsed/bs:.3f}s/item, avg_output_tokens={avg_out_len:.0f}")


def profile_end_to_end(model, tokenizer, model_name, device, eval_dir, cache_dir, stride, act_layers, max_items=20, eval_batch_size=8):
    """Profile a full training eval run with per-phase timing."""
    print(f"\n{'='*60}")
    print(f"END-TO-END EVAL PROFILING")
    print(f"  max_items={max_items}, eval_batch_size={eval_batch_size}")
    print(f"  cache_dir={cache_dir}")
    print(f"{'='*60}")

    set_oracle_mode(trained=True, oracle_adapter_name="default", stride=stride, layers=act_layers)

    eval_list = ["hinted_mcq_truthfulqa", "sycophancy_v2_riya", "atypical_answer_riya"]

    for eval_name in eval_list:
        print(f"\n--- {eval_name} ---")
        items = load_eval_items_hf(eval_name, eval_dir=str(eval_dir))
        items = _subsample(items, max_items, seed=hash(eval_name))
        print(f"  {len(items)} items after subsample")

        # Phase 1: Activation extraction (per-item, sequential)
        extracted = []
        t_extract_total = 0.0
        n_cached = 0
        n_fresh = 0

        for item in items:
            cached = _try_load_cached(cache_dir, eval_name, item.example_id, device)
            if cached is not None:
                n_cached += 1
                clean_response = cached.clean_response or ""
                test_response = cached.test_response or ""
                clean_answer = cached.clean_answer or _extract_answer(clean_response, eval_name)
                test_answer = cached.test_answer or _extract_answer(test_response, eval_name)
                activations = cached.activations
            else:
                precomp_test = _get_test_response(item)
                precomp_clean = item.metadata.get("qwen3_8b_clean_response")
                if not precomp_clean and precomp_test and item.clean_prompt == item.test_prompt:
                    precomp_clean = precomp_test
                clean_response = precomp_clean or ""
                test_response = precomp_test or ""
                clean_answer = _extract_answer(clean_response, eval_name)
                test_answer = _extract_answer(test_response, eval_name)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                bundle = extract_activations(
                    model, tokenizer,
                    eval_name=eval_name, example_id=item.example_id,
                    prompt=item.test_prompt, cot_text=test_response.strip(),
                    stride=stride, layers=act_layers, device=device,
                    generation_adapter_name=None,
                )
                torch.cuda.synchronize()
                t_extract_total += time.perf_counter() - t0
                n_fresh += 1
                activations = bundle.activations if bundle else None

            oracle_prompt = ""
            if activations is not None:
                template = ORACLE_PROMPTS_TEMPLATES.get(eval_name, "What is this model doing?")
                oracle_prompt = _oracle_prompt(activations.shape[0], template)

            ground_truth = determine_ground_truth(item, clean_answer, test_answer)
            extracted.append({
                "item": item, "activations": activations,
                "oracle_prompt": oracle_prompt, "oracle_response": "",
                "clean_response": clean_response, "test_response": test_response,
                "clean_answer": clean_answer, "test_answer": test_answer,
                "ground_truth": ground_truth,
            })

        print(f"  Phase 1 (extraction): {t_extract_total:.3f}s ({n_cached} cached, {n_fresh} fresh)")
        if n_fresh > 0:
            print(f"    {t_extract_total/n_fresh:.3f}s per fresh item")

        # Phase 2: Oracle generation
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _collect_and_batch_oracle(
            extracted, model, tokenizer, model_name, device,
            eval_name=eval_name, eval_batch_size=eval_batch_size,
        )
        torch.cuda.synchronize()
        t_oracle = time.perf_counter() - t0
        n_with_acts = sum(1 for ex in extracted if ex["activations"] is not None)
        print(f"  Phase 2 (oracle gen): {t_oracle:.3f}s ({n_with_acts} items with activations)")
        if n_with_acts > 0:
            print(f"    {t_oracle/n_with_acts:.3f}s per item")

        # Show a sample response
        for ex in extracted:
            if ex["oracle_response"]:
                resp = ex["oracle_response"].strip()[:200]
                print(f"  Sample oracle response: '{resp}'")
                break

        print(f"  TOTAL: {t_extract_total + t_oracle:.3f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (PeftModel)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--eval-dir", type=str, default="data/evals")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--profile", choices=["extraction", "oracle", "e2e", "all"], default="all")
    args = parser.parse_args()

    device = "cuda"
    act_layers = get_injection_layers(args.model_name)
    print(f"Injection layers: {act_layers}")
    print(f"Stride: {args.stride}")

    # Resolve cache dir
    cache_dir = args.cache_dir
    if cache_dir is None:
        fast_cache = os.environ.get("FAST_CACHE_DIR")
        if fast_cache:
            cache_dir = os.path.join(fast_cache, "cot_oracle", "eval_precomputed")
        else:
            cache_dir = "data/eval_precomputed"
    cache_dir = Path(cache_dir)
    print(f"Cache dir: {cache_dir} (exists={cache_dir.exists()})")

    eval_dir = Path(args.eval_dir)

    # Load model
    print(f"\nLoading model: {args.model_name}")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )

    if args.checkpoint:
        print(f"Loading PeftModel from {args.checkpoint}")
        model = PeftModel.from_pretrained(base_model, args.checkpoint, adapter_name="default")
    else:
        # Load original AO checkpoint for oracle
        ao_ckpt = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
        print(f"Loading AO adapter: {ao_ckpt}")
        model = PeftModel.from_pretrained(base_model, ao_ckpt, adapter_name="default")

    model.eval()
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # Set oracle mode
    set_oracle_mode(trained=True, oracle_adapter_name="default", stride=args.stride, layers=act_layers)

    # Run profiles
    if args.profile in ("extraction", "all"):
        items = load_eval_items_hf("hinted_mcq_truthfulqa", eval_dir=str(eval_dir))
        items = _subsample(items, args.max_items, seed=hash("hinted_mcq_truthfulqa"))
        profile_activation_extraction(model, tokenizer, items, "hinted_mcq_truthfulqa", act_layers, device, args.stride)

    if args.profile in ("oracle", "all"):
        items = load_eval_items_hf("hinted_mcq_truthfulqa", eval_dir=str(eval_dir))
        items = _subsample(items, args.max_items, seed=hash("hinted_mcq_truthfulqa"))
        profile_oracle_generation(model, tokenizer, items, "hinted_mcq_truthfulqa", act_layers, args.model_name, device, args.stride, cache_dir)

    if args.profile in ("e2e", "all"):
        profile_end_to_end(
            model, tokenizer, args.model_name, device, eval_dir, cache_dir,
            args.stride, act_layers, max_items=args.max_items, eval_batch_size=args.eval_batch_size,
        )

    print("\n\nDONE.")


if __name__ == "__main__":
    main()

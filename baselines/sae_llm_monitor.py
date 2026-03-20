"""Baseline: SAE feature description → LLM judge.

Encodes activations through BatchTopK SAEs, aggregates top-K features with
human-readable labels, then sends feature descriptions to an LLM to answer
the eval question.

Unified API: accepts test_data + activations from eval_comprehensive.
"""

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import openai
import torch
from tqdm.auto import tqdm

_AO = Path(__file__).resolve().parent.parent / "ao_reference"
if str(_AO) not in sys.path:
    sys.path.insert(0, str(_AO))

from nl_probes.sae import load_dictionary_learning_batch_topk_sae
from activation_utils import split_activations_by_layer

SAE_REPO = "adamkarvonen/qwen3-8b-saes"
MODEL_NAME = "Qwen/Qwen3-8B"

BINARY_PROMPT = (
    "{cot_preamble} You are analyzing its internal representations via SAE features.\n\n"
    "The following sparse autoencoder features were most active in the model's residual stream:\n\n"
    "{feature_desc}\n\n"
    "Based on these features, answer the following question about the reasoning:\n"
    "{eval_question}\n\n"
    "Answer with ONLY one word: {option_a} or {option_b}."
)

GENERATION_PROMPT = (
    "{cot_preamble} You are analyzing its internal representations via SAE features.\n\n"
    "The following sparse autoencoder features were most active in the residual stream:\n\n"
    "{feature_desc}\n\n"
    "Based on these features:\n{eval_question}\n\n"
    "Give a concise answer."
)


def _load_saes(sae_dir: str, trainer: int, layers: list[int], device: str) -> dict:
    saes = {}
    for layer in layers:
        filename = f"saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_{layer}/trainer_{trainer}/ae.pt"
        print(f"  Loading SAE layer {layer}...")
        saes[layer] = load_dictionary_learning_batch_topk_sae(
            repo_id=SAE_REPO, filename=filename, model_name=MODEL_NAME,
            device=torch.device(device), dtype=torch.bfloat16, layer=layer,
            local_dir=sae_dir,
        )
    return saes


def _load_labels(labels_dir: str, trainer: int, layers: list[int]) -> dict:
    labels = {}
    for layer in layers:
        path = Path(labels_dir) / f"labels_layer{layer}_trainer{trainer}.json"
        print(f"  Loading SAE labels: {path}")
        labels[layer] = json.loads(path.read_text())
    return labels


def _encode_and_aggregate(saes, labels, acts_by_layer, layers, top_k):
    aggregated = {}
    for layer in layers:
        if layer not in acts_by_layer or layer not in saes:
            continue
        sae = saes[layer]
        layer_labels = labels.get(layer, {})
        acts = acts_by_layer[layer]

        sae_acts = sae.encode(acts.to(sae.W_enc.device))

        feat_counts = Counter()
        feat_sum_act = Counter()
        feat_labels = {}
        for i in range(sae_acts.shape[0]):
            topk = sae_acts[i].topk(min(top_k, sae_acts.shape[1]))
            for val, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                if val <= 0:
                    continue
                feat_counts[idx] += 1
                feat_sum_act[idx] += val
                label_info = layer_labels.get(str(idx), {})
                feat_labels[idx] = label_info.get("label", "unlabeled")

        ranked = []
        for feat_idx, count in feat_counts.most_common(top_k):
            mean_act = feat_sum_act[feat_idx] / count
            ranked.append((feat_idx, count, mean_act, feat_labels[feat_idx]))
        aggregated[layer] = ranked
    return aggregated


def _format_features(aggregated, n_positions):
    parts = []
    for layer in sorted(aggregated.keys()):
        features = aggregated[layer]
        if not features:
            continue
        lines = [f"Layer {layer} (top features by frequency across {n_positions} positions):"]
        for feat_idx, count, mean_act, label in features:
            lines.append(f"  - Feature {feat_idx} (freq={count}, strength={mean_act:.1f}): {label}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _build_sae_prompt(feature_desc, item, task_def):
    prompt_text = item.get("prompt", "")
    preamble = task_def.cot_preamble
    if task_def.positive_label and task_def.negative_label:
        return BINARY_PROMPT.format(
            cot_preamble=preamble,
            feature_desc=feature_desc,
            eval_question=prompt_text,
            option_a=task_def.positive_label,
            option_b=task_def.negative_label,
        )
    return GENERATION_PROMPT.format(
        cot_preamble=preamble,
        feature_desc=feature_desc,
        eval_question=f"Question: {prompt_text[:2000]}",
    )


async def _fetch_one(client, sem, prompt, model, max_tokens, temperature, max_retries=5):
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, temperature=temperature,
                )
                return response.choices[0].message.content or ""
            except openai.RateLimitError:
                await asyncio.sleep(2 ** attempt + 1)
        response = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature,
        )
        return response.choices[0].message.content or ""


async def _fetch_batch(client, sem, prompts, model, max_tokens, temperature, pbar):
    async def _wrapped(prompt):
        result = await _fetch_one(client, sem, prompt, model, max_tokens, temperature)
        pbar.update(1)
        return result
    return await asyncio.gather(*[_wrapped(p) for p in prompts])


def run_sae_probe(
    test_data: list[dict],
    activations: list[torch.Tensor],
    layers: list[int],
    task_def,
    *,
    top_k: int = 20,
    sae_dir: str = "downloaded_saes",
    sae_labels_dir: str = "",
    sae_trainer: int = 2,
    llm_model: str = "google/gemini-2.5-flash-lite",
    api_key: str = "",
    api_base: str = "https://openrouter.ai/api/v1",
    max_tokens: int = 300,
    temperature: float = 0.0,
    device: str = "cuda",
    max_concurrent: int = 20,
) -> tuple[list[str], list[str]]:
    """Run SAE probe. Returns (responses, prompts)."""
    saes = _load_saes(sae_dir, sae_trainer, layers, device)
    sae_labels = _load_labels(sae_labels_dir, sae_trainer, layers)

    prompts = []
    for item, acts in tqdm(zip(test_data, activations), desc="SAE encoding", total=len(test_data)):
        acts_by_layer = split_activations_by_layer(acts, layers)
        aggregated = _encode_and_aggregate(saes, sae_labels, acts_by_layer, layers, top_k)
        n_positions = max((acts_by_layer[l].shape[0] for l in layers if l in acts_by_layer), default=0)
        feature_desc = _format_features(aggregated, n_positions)
        prompts.append(_build_sae_prompt(feature_desc, item, task_def))

    async def _run_all():
        client = openai.AsyncOpenAI(base_url=api_base, api_key=api_key)
        sem = asyncio.Semaphore(max_concurrent)
        try:
            return await _fetch_batch(client, sem, prompts, llm_model, max_tokens, temperature, pbar)
        finally:
            await client.close()

    pbar = tqdm(total=len(prompts), desc="SAE probe (API)")
    responses = asyncio.run(_run_all())
    pbar.close()

    return responses, prompts

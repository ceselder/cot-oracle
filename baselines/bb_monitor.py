"""Baseline: External black-box monitor via OpenRouter (text-only, zero-shot).

Unified API: accepts test_data from eval_comprehensive.
Supports incremental caching via EvalCache.
"""

import os
import re

import httpx
from tqdm.auto import tqdm

_MONITOR_TEMPLATE = (
    "{cot_preamble}\n\n"
    "Question: {question}\n\n"
    "Reasoning:\n{cot}\n\n"
    "Task: {prompt}"
)


def run_bb_monitor(
    test_data: list[dict],
    activations: list,  # unused (text-only), kept for signature consistency
    layers: list[int],
    task_def,
    *,
    model_name: str = "",
    max_tokens: int = 300,
    temperature: float = 0.0,
    cache=None,
    run_id: str | None = None,
    method_name: str | None = None,
    completed_indices: set | None = None,
) -> tuple[list[str | None], list[str | None]]:
    """Run BB monitor. Returns (predictions, prompts) — both per-item lists."""
    cot_field = task_def.cot_field
    prompt_template = _MONITOR_TEMPLATE.replace("{cot_preamble}", task_def.cot_preamble)

    test_data = [d for d in test_data if d.get(cot_field)]
    if not test_data:
        return [], []

    completed = completed_indices or set()
    todo_indices = [i for i in range(len(test_data)) if i not in completed]
    if not todo_indices:
        return [None] * len(test_data), [None] * len(test_data)

    api_key = os.environ["OPENROUTER_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    predictions = [None] * len(test_data)
    prompts = [None] * len(test_data)

    with httpx.Client(timeout=120.0) as client:
        for i in tqdm(todo_indices, desc=f"BB monitor ({model_name.split('/')[-1]})", leave=False):
            item = test_data[i]
            user_msg = prompt_template.format(
                question=item.get("question", ""),
                cot=item.get(cot_field, ""),
                prompt=item.get("prompt", ""),
            )
            prompts[i] = user_msg

            body = {
                "model": model_name,
                "messages": [{"role": "user", "content": user_msg}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            response = client.post(
                f"{os.environ.get('OPENROUTER_API_BASE', 'https://openrouter.ai/api/v1').rstrip('/')}/chat/completions",
                json=body, headers=headers,
            )
            response.raise_for_status()
            raw_text = (response.json()["choices"][0]["message"]["content"] or "").strip()
            raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            predictions[i] = raw_text

            # Store incrementally
            if cache is not None:
                cache.store_predictions(run_id, item.get("_task_name", ""), method_name, [{
                    "item_idx": i, "item_id": f"{item.get('_task_name', '')}_{i}",
                    "prediction": raw_text[:500], "score": None,
                    "prompt": user_msg[:2000],
                }])

    return predictions, prompts

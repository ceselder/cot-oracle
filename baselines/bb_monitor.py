"""Baseline: External black-box monitor via OpenRouter (text-only, zero-shot).

Unified API: accepts test_data from eval_comprehensive.
Supports incremental caching via EvalCache.
"""

import asyncio
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


async def _fetch_one(client, url, headers, body, sem, max_retries=3):
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.post(url, json=body, headers=headers)
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                raw = (resp.json()["choices"][0]["message"]["content"] or "").strip()
                return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt * 5, 60))
                    continue
                raise
    return ""


def run_bb_monitor(
    test_data: list[dict],
    activations: list,  # unused (text-only), kept for signature consistency
    layers: list[int],
    task_def,
    *,
    model_name: str = "",
    max_tokens: int = 300,
    temperature: float = 0.0,
    max_concurrent: int = 20,
    cache=None,
    run_id: str | None = None,
    method_name: str | None = None,
    completed_indices: set | None = None,
) -> tuple[list[str | None], list[str | None]]:
    """Run BB monitor. Returns (predictions, prompts) — both per-item lists."""
    supervisor_context = task_def.supervisor_context
    prompt_template = _MONITOR_TEMPLATE.replace("{cot_preamble}", task_def.cot_preamble)

    test_data = [d for d in test_data if d.get(supervisor_context)]
    if not test_data:
        return [], []

    completed = completed_indices or set()
    todo_indices = [i for i in range(len(test_data)) if i not in completed]
    if not todo_indices:
        return [None] * len(test_data), [None] * len(test_data)

    api_key = os.environ["OPENROUTER_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{os.environ.get('OPENROUTER_API_BASE', 'https://openrouter.ai/api/v1').rstrip('/')}/chat/completions"

    predictions = [None] * len(test_data)
    prompts = [None] * len(test_data)

    # Build all requests
    requests = []  # (index, user_msg, body)
    for i in todo_indices:
        item = test_data[i]
        user_msg = prompt_template.format(
            question=item.get("question", ""),
            cot=item.get(supervisor_context, ""),
            prompt=item.get("prompt", ""),
        )
        prompts[i] = user_msg
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": user_msg}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        requests.append((i, user_msg, body))

    async def _run_all():
        sem = asyncio.Semaphore(max_concurrent)
        async with httpx.AsyncClient(timeout=120.0) as client:
            tasks = [_fetch_one(client, url, headers, body, sem) for _, _, body in requests]
            return await asyncio.gather(*tasks, return_exceptions=True)

    pbar = tqdm(total=len(requests), desc=f"BB monitor ({model_name.split('/')[-1]})", leave=False)
    results = asyncio.run(_run_all())

    for (i, user_msg, _), result in zip(requests, results):
        pbar.update(1)
        if isinstance(result, Exception):
            print(f"    [bb_monitor] item {i} failed: {result}")
            continue
        predictions[i] = result

        if cache is not None:
            item = test_data[i]
            cache.store_predictions(run_id, item.get("_task_name", ""), method_name, [{
                "item_idx": i, "item_id": f"{item.get('_task_name', '')}_{i}",
                "prediction": result, "score": None,
                "method_prompt": user_msg,
            }])

    pbar.close()
    return predictions, prompts

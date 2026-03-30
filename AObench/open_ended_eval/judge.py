"""
Shared LLM judge for AObench evals.

Uses a local Sonnet wrapper (free, via Claude subscription) by default,
with OpenRouter as fallback.
"""

import asyncio
import json
import os
from typing import Any

import httpx

# Local free Sonnet endpoint (Claude subscription wrapper on Hetzner box)
LOCAL_API_BASE = "http://95.216.187.49:8765/v1/chat/completions"
LOCAL_API_KEY = "cot-oracle-judge-2026"

# OpenRouter fallback
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

JUDGE_MODEL = "claude-sonnet-4-6"
DEFAULT_JUDGE_CONCURRENCY = 10  # lower for local wrapper rate limits


def _get_endpoint() -> tuple[str, str]:
    """Return (api_base, api_key) for the judge endpoint."""
    use_local = os.environ.get("JUDGE_USE_LOCAL", "1") != "0"
    if use_local:
        return LOCAL_API_BASE, LOCAL_API_KEY
    # OpenRouter fallback
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set and local judge unavailable.")
    return OPENROUTER_API_BASE, key


async def judge_single(
    client: httpx.AsyncClient,
    system_prompt: str,
    user_message: str,
    semaphore: asyncio.Semaphore,
    model: str = JUDGE_MODEL,
    max_tokens: int = 300,
) -> dict[str, Any]:
    """Call judge endpoint and parse JSON response."""
    api_base, api_key = _get_endpoint()

    # Local wrapper doesn't support system messages — fold into user message
    use_local = os.environ.get("JUDGE_USE_LOCAL", "1") != "0"
    if use_local:
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{user_message}"},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    async with semaphore:
        resp = await client.post(
            api_base,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            },
            timeout=120.0,
        )
        resp.raise_for_status()

    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
        text = text.strip()

    result = json.loads(text)
    return result

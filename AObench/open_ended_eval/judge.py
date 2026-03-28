"""
Shared LLM judge via OpenRouter for AObench evals.

Uses httpx async client to call OpenRouter's chat completions API.
"""

import asyncio
import json
import os
from typing import Any

import httpx

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_JUDGE_CONCURRENCY = 20


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Required for LLM judge evals."
        )
    return key


async def judge_single(
    client: httpx.AsyncClient,
    system_prompt: str,
    user_message: str,
    semaphore: asyncio.Semaphore,
    model: str = JUDGE_MODEL,
    max_tokens: int = 300,
) -> dict[str, Any]:
    """Call OpenRouter and parse JSON response."""
    api_key = _get_api_key()

    async with semaphore:
        resp = await client.post(
            OPENROUTER_API_BASE,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            },
            timeout=60.0,
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

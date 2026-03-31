"""
Shared LLM judge for AObench evals.

Uses a local Sonnet wrapper (free, via Claude subscription) by default,
with OpenRouter as fallback.
"""

import asyncio
import json
import os
import re
from typing import Any

import httpx

# Local free Sonnet endpoint (Claude subscription wrapper on Hetzner box)
LOCAL_API_BASE = "http://95.216.187.49:8765/v1/chat/completions"
LOCAL_API_KEY = "cot-oracle-judge-2026"

# OpenRouter fallback
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
DEFAULT_JUDGE_CONCURRENCY = 10  # lower for local wrapper rate limits


def _extract_json_payload(text: str) -> dict[str, Any]:
    """Parse a JSON object from judge text with light wrapper tolerance."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return json.loads(match.group(0))


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
    model: str | None = None,
    max_tokens: int = 120,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call judge endpoint and parse JSON response."""
    api_base, api_key = _get_endpoint()
    if model is None:
        model = os.environ.get("JUDGE_MODEL", JUDGE_MODEL)

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

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
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
            return _extract_json_payload(text)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            await asyncio.sleep(1.0 * (attempt + 1))

    assert last_error is not None
    raise last_error

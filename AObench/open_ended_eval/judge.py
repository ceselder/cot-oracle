"""
Shared LLM judge for AObench evals.

Uses a local Sonnet wrapper (free, via Claude subscription) by default and,
when enabled, falls back to OpenRouter Sonnet on every failed local attempt.
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
DEFAULT_JUDGE_CONCURRENCY = int(os.environ.get("JUDGE_CONCURRENCY", "10"))
DEFAULT_OPENROUTER_FALLBACK_MODEL = os.environ.get(
    "OPENROUTER_JUDGE_FALLBACK_MODEL",
    "anthropic/claude-sonnet-4.6",
)


def _extract_json_payload(text: str | None) -> dict[str, Any]:
    """Parse a JSON object from judge text with light wrapper tolerance."""
    if text is None:
        raise json.JSONDecodeError("Judge response content was null", "", 0)

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise json.JSONDecodeError("Parsed JSON payload is not an object", text, 0)
        return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("Parsed JSON payload is not an object", match.group(0), 0)
    return payload


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


def _openrouter_fallback_model(model: str) -> str:
    """Map local wrapper aliases to a concrete OpenRouter Sonnet model."""
    if model.startswith("anthropic/"):
        return model

    alias_map = {
        "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
        "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
        "claude-sonnet-4-5": "anthropic/claude-sonnet-4.5",
        "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
        "claude-sonnet-4": "anthropic/claude-sonnet-4",
        "claude-sonnet-4.0": "anthropic/claude-sonnet-4",
    }
    return alias_map.get(model, DEFAULT_OPENROUTER_FALLBACK_MODEL)


async def _call_judge_endpoint(
    *,
    client: httpx.AsyncClient,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
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
    text = data["choices"][0]["message"]["content"]
    return _extract_json_payload(text)


async def judge_single(
    client: httpx.AsyncClient,
    system_prompt: str,
    user_message: str,
    semaphore: asyncio.Semaphore,
    model: str | None = None,
    max_tokens: int = 120,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call judge endpoint and parse JSON response.

    When `JUDGE_USE_LOCAL=1`, each retry cycle attempts the local Sonnet
    wrapper first, then falls back to OpenRouter Sonnet if the local call or
    parse fails.
    """
    if model is None:
        model = os.environ.get("JUDGE_MODEL", JUDGE_MODEL)

    use_local = os.environ.get("JUDGE_USE_LOCAL", "1") != "0"
    local_messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{user_message}"},
    ]
    openrouter_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries):
        if use_local:
            try:
                return await _call_judge_endpoint(
                    client=client,
                    api_base=LOCAL_API_BASE,
                    api_key=LOCAL_API_KEY,
                    model=model,
                    messages=local_messages,
                    max_tokens=max_tokens,
                    semaphore=semaphore,
                )
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc

        try:
            _, api_key = _get_endpoint() if not use_local else (OPENROUTER_API_BASE, os.environ.get("OPENROUTER_API_KEY", ""))
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set for judge fallback.")
            return await _call_judge_endpoint(
                client=client,
                api_base=OPENROUTER_API_BASE,
                api_key=api_key,
                model=_openrouter_fallback_model(model),
                messages=openrouter_messages,
                max_tokens=max_tokens,
                semaphore=semaphore,
            )
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            await asyncio.sleep(1.0 * (attempt + 1))

    assert last_error is not None
    raise last_error

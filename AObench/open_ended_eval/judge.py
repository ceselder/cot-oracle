"""
Shared LLM judge for AObench evals.

Endpoint priority (configurable via env):
1. Native Anthropic API (when ANTHROPIC_API_KEY is set and JUDGE_USE_ANTHROPIC=1)
   — preferred: cheaper, supports prompt caching, first-class features.
   On persistent rate-limits, swaps to ANTHROPIC_API_KEY_FALLBACK.
2. Local Sonnet wrapper (free Hetzner box) when JUDGE_USE_LOCAL=1.
3. OpenRouter fallback when OPENROUTER_API_KEY is set.
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

# Native Anthropic
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
DEFAULT_JUDGE_CONCURRENCY = int(os.environ.get("JUDGE_CONCURRENCY", "10"))
DEFAULT_OPENROUTER_FALLBACK_MODEL = os.environ.get(
    "OPENROUTER_JUDGE_FALLBACK_MODEL",
    "anthropic/claude-sonnet-4.6",
)


def _use_anthropic() -> bool:
    return os.environ.get("JUDGE_USE_ANTHROPIC", "1") != "0" and bool(
        os.environ.get("ANTHROPIC_API_KEY")
    )


def _use_local() -> bool:
    # Default off when Anthropic native is available, else on for back-compat.
    if _use_anthropic():
        return os.environ.get("JUDGE_USE_LOCAL", "0") != "0"
    return os.environ.get("JUDGE_USE_LOCAL", "1") != "0"


def _anthropic_model(model: str) -> str:
    """Strip OpenRouter prefix if present so the same JUDGE_MODEL works for both."""
    if model.startswith("anthropic/"):
        return model.split("/", 1)[1].replace("claude-sonnet-4.6", "claude-sonnet-4-6")
    return model


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


async def _call_anthropic_native(
    *,
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Call Anthropic native /v1/messages with prompt caching on the system block."""
    async with semaphore:
        resp = await client.post(
            ANTHROPIC_API_BASE,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json={
                "model": _anthropic_model(model),
                "max_tokens": max_tokens,
                "system": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [{"role": "user", "content": user_message}],
            },
            timeout=120.0,
        )
        resp.raise_for_status()
    data = resp.json()
    text = data["content"][0]["text"]
    return _extract_json_payload(text)


async def _call_chat_completions(
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

    Order on each attempt: native Anthropic (low-prio key, then fallback key)
    -> local Sonnet wrapper -> OpenRouter Sonnet.
    """
    if model is None:
        model = os.environ.get("JUDGE_MODEL", JUDGE_MODEL)

    use_anthropic = _use_anthropic()
    use_local = _use_local()

    anthropic_keys: list[str] = []
    if use_anthropic:
        primary = os.environ.get("ANTHROPIC_API_KEY", "")
        if primary:
            anthropic_keys.append(primary)
        fallback = os.environ.get("ANTHROPIC_API_KEY_FALLBACK", "")
        if fallback and fallback != primary:
            anthropic_keys.append(fallback)

    local_messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{user_message}"},
    ]
    openrouter_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries):
        for ant_key in anthropic_keys:
            try:
                return await _call_anthropic_native(
                    client=client,
                    api_key=ant_key,
                    model=model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    max_tokens=max_tokens,
                    semaphore=semaphore,
                )
            except (
                httpx.HTTPError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                last_error = exc

        if use_local:
            try:
                return await _call_chat_completions(
                    client=client,
                    api_base=LOCAL_API_BASE,
                    api_key=LOCAL_API_KEY,
                    model=model,
                    messages=local_messages,
                    max_tokens=max_tokens,
                    semaphore=semaphore,
                )
            except (
                httpx.HTTPError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                last_error = exc

        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        if openrouter_key:
            try:
                return await _call_chat_completions(
                    client=client,
                    api_base=OPENROUTER_API_BASE,
                    api_key=openrouter_key,
                    model=_openrouter_fallback_model(model),
                    messages=openrouter_messages,
                    max_tokens=max_tokens,
                    semaphore=semaphore,
                )
            except (
                httpx.HTTPError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as exc:
                last_error = exc

        if attempt == max_retries - 1:
            break
        await asyncio.sleep(1.0 * (attempt + 1))

    if last_error is None:
        last_error = RuntimeError(
            "No judge endpoint configured: set ANTHROPIC_API_KEY, "
            "JUDGE_USE_LOCAL=1, or OPENROUTER_API_KEY."
        )
    raise last_error

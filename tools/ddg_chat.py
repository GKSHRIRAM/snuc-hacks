"""
DuckDuckGo AI Chat client — free, no API key required.
Uses GPT-4o-mini via DDG's undocumented but stable duckchat API.

Flow:
  1. GET /duckchat/v1/status  → returns x-vqd-4 token
  2. POST /duckchat/v1/chat   → SSE stream of delta tokens
  3. Collect deltas → return full string
"""

import httpx
import json
import asyncio

DDG_STATUS_URL = "https://duckduckgo.com/duckchat/v1/status"
DDG_CHAT_URL   = "https://duckduckgo.com/duckchat/v1/chat"
DDG_MODEL      = "gpt-4o-mini"   # free tier model; also: "claude-3-haiku-20240307"

HEADERS_BASE = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/event-stream",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://duckduckgo.com/",
    "Origin": "https://duckduckgo.com",
    "x-vqd-accept": "1",
}


async def _get_vqd_token(client: httpx.AsyncClient) -> str:
    """Fetch the one-time VQD token required for DDG chat."""
    resp = await client.get(DDG_STATUS_URL, headers=HEADERS_BASE, timeout=15.0)
    resp.raise_for_status()
    token = resp.headers.get("x-vqd-4")
    if not token:
        raise RuntimeError(f"DDG chat: no x-vqd-4 token in status response. Headers: {dict(resp.headers)}")
    return token


async def ddg_chat(messages: list, temperature: float = 0.0, max_retries: int = 3) -> str:
    """
    Send a list of OpenAI-style messages to DDG AI Chat and return the full reply.
    Retries on transient errors with exponential backoff.
    temperature is accepted for API-compat but DDG ignores it.
    """
    last_error: Exception = RuntimeError("No attempts made")

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                token = await _get_vqd_token(client)

                headers = {**HEADERS_BASE, "x-vqd-4": token, "Content-Type": "application/json"}
                payload = {"model": DDG_MODEL, "messages": messages}

                full_text = ""
                async with client.stream(
                    "POST", DDG_CHAT_URL, headers=headers, json=payload, timeout=90.0
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("message", "")
                            full_text += delta
                        except json.JSONDecodeError:
                            continue

                return full_text.strip()

        except (httpx.HTTPStatusError, httpx.TimeoutException, RuntimeError) as e:
            last_error = e
            wait = 2 ** attempt
            print(f"  DDG chat attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {str(e)[:80]}. Retrying in {wait}s...")
            await asyncio.sleep(wait)

    raise RuntimeError(f"DDG chat failed after {max_retries} retries. Last error: {last_error}")

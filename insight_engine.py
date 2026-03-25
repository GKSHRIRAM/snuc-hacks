import os
import json
import httpx
import asyncio
import logging
from typing import Optional
from differ import ExportDiff

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

GROQ_MODEL   = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


async def _groq_chat(messages: list, max_tokens: int = 1024, max_retries: int = 4) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "messages": messages,
               "temperature": 0.2, "max_tokens": max_tokens}
    for attempt in range(max_retries):
        async with httpx.AsyncClient() as client:
            r = await client.post(GROQ_API_URL, headers=headers, json=payload, timeout=60.0)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Groq rate-limited. Waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                await asyncio.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError(f"Groq API failed after {max_retries} retries (persistent 429).")


def build_insight_prompt(diff: ExportDiff) -> str:
    n = len(diff.competitor_diffs)
    parts: list[str] = []

    parts.append(
        f"You are an elite competitive intelligence analyst. The startup is exploring the "
        f"'{diff.industry}' market. Below is a diff of {n} competitors: historical vs live.\n"
    )

    for comp in diff.competitor_diffs:
        h_date_str: str = comp.historical_date.strftime("%Y-%m-%d") if comp.historical_date else "Unknown"
        l_date_str: str = comp.live_date.strftime("%Y-%m-%d") if comp.live_date else "Unknown"

        pricing_parts: list[str] = []
        for p in comp.pricing_changes:
            desc: str = "{}: {}".format(p.tier_name, p.change_type.replace("_", " ").title())
            if p.delta_usd is not None:
                desc = desc + " (Delta: ${:.2f})".format(p.delta_usd)
            elif p.historical_price is not None and p.live_price is not None:
                desc = desc + " (${:.2f} -> ${:.2f})".format(p.historical_price, p.live_price)
            elif p.live_price is not None:
                desc = desc + " (${:.2f})".format(p.live_price)
            pricing_parts.append(desc)

        pricing_text: str = ", ".join(pricing_parts) if pricing_parts else "None detected"
        features_added: str = ", ".join(comp.features_added[:3]) if comp.features_added else "None"
        features_removed: str = ", ".join(comp.features_removed[:3]) if comp.features_removed else "None"
        score_delta_str: str = (
            str(round(comp.sentiment_score_delta, 2)) if comp.sentiment_score_delta is not None else "N/A"
        )
        new_complaints: str = ", ".join(comp.new_complaints[:2]) if comp.new_complaints else "None"

        parts.append(
            "### {}: {} -> {}\n".format(comp.competitor_name, h_date_str, l_date_str)
            + "Pricing: {} | Added: {} | Removed: {} | Sentiment: {} | Complaints: {}\n".format(
                pricing_text, features_added, features_removed, score_delta_str, new_complaints
            )
            + "Summary: {}\n".format(comp.programmatic_summary)
        )

    parts.append(
        'Respond with valid JSON only (no markdown fences):\n'
        '{"momentum_leader":"...","pricing_patterns":["..."],"opportunity_gaps":["..."],'
        '"sentiment_insights":["..."],"competitor_summaries":{"name":"..."},'
        '"full_analysis":"...","sources":[]}'
    )

    return "\n".join(parts)


async def get_insights(diff: ExportDiff) -> dict:
    """Call Groq to generate competitive insights from the diff."""
    prompt = build_insight_prompt(diff)

    print("  Calling Groq for insights...")
    raw = await _groq_chat([
        {"role": "system", "content": "You are a competitive intelligence analyst. Respond with valid JSON only — no markdown, no preamble."},
        {"role": "user", "content": prompt}
    ], max_tokens=1500)

    content = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Groq returned invalid JSON. Error: {e}\nRaw: {content[:500]}")


def save_insights(insights: dict, diff: ExportDiff) -> str:
    """Save insights JSON to data_exports/insights/ and return the file path."""
    from datetime import datetime
    os.makedirs("data_exports/insights", exist_ok=True)
    ts = diff.export_timestamp.strftime("%Y%m%d_%H%M%S") if diff.export_timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    company = diff.startup_query.replace(" ", "_").lower() if diff.startup_query else "analysis"
    path = os.path.join("data_exports/insights", f"{company}_{ts}_insights.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)
    print(f"  Insights saved to {path}")
    return path

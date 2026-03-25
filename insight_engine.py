import os
import json
import httpx
import asyncio
import logging
from typing import Optional
from differ import ExportDiff

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from tools.llm_client import llm_chat_json

async def get_insights(diff: ExportDiff) -> dict:
    """Call Local LLM to generate competitive insights from the diff."""
    prompt = build_insight_prompt(diff)

    print("  Calling Local LLM for insights...")
    try:
        return await llm_chat_json([
            {"role": "system", "content": "You are a competitive intelligence analyst. Respond with valid JSON only — no markdown, no preamble."},
            {"role": "user", "content": prompt}
        ])
    except Exception as e:
        print(f"  [Insights] Failed: {e}")
        raise RuntimeError(f"Local LLM failed to generate valid insights: {e}")


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

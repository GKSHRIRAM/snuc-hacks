import os
import json
from dotenv import load_dotenv

load_dotenv()

import asyncio
import httpx
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tools.searxng import search_competitors_with_searxng
from tools.wayback import get_wayback_snapshot
from tools.firecrawl_extractor import extract_markdown_with_firecrawl
from tools.wayback_archiver import archive_to_wayback
from tools.reddit_scraper import get_reddit_sentiment_sync

app = FastAPI(
    title="MarketLens BI Engine",
    description="Zero-storage market intelligence pipeline with industry-constrained discovery.",
    version="3.1.0"
)

class AnalyzeRequest(BaseModel):
    user_prompt: str

# ══════════════════════════════════════════════
# Phase 0: Understanding Agent
# ══════════════════════════════════════════════
async def run_understanding_agent(prompt: str) -> Dict[str, str]:
    """Identifies industry and lists 5 real competitors (incumbents + startups)."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_key_here":
        raise ValueError("GROQ_API_KEY is missing.")

    system_prompt = """You are a market research analyst. The user will describe their startup idea.
Your job is to:
1. Identify the industry/domain.
2. Name the top 5 existing real-world competitors — include BOTH major incumbents AND newer startups/challengers.
   IMPORTANT: Every competitor you name MUST be a software/tech company in the SAME industry as the user's startup.
   Do NOT name companies from unrelated industries.

Respond in this EXACT format (no extra text):
industry: <industry name>
competitor_1: <biggest incumbent — full company name>
competitor_2: <second biggest — full company name>
competitor_3: <third company — full company name>
competitor_4: <emerging startup or challenger — full company name>
competitor_5: <another startup or niche player — full company name>"""

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload, timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            raw = data["choices"][0]["message"]["content"].strip()

        result = {}
        for line in raw.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                result[k.strip()] = v.strip()
        return result
    except Exception as e:
        print(f"Understanding Agent Error: {e}")
        return {"industry": "unknown"}

# ══════════════════════════════════════════════
# Phase 1: Discovery Agent (Industry-Constrained)
# ══════════════════════════════════════════════
async def run_discovery_agent(understanding: Dict[str, str]) -> Dict[str, str]:
    """Discovers actual URLs. Injects industry into every search to prevent off-topic results."""
    industry = understanding.get("industry", "software")
    url_map = {}
    for i in range(1, 6):
        name = understanding.get(f"competitor_{i}", "")
        if not name:
            continue
        query = f"{name} {industry} software official site pricing"
        urls = await search_competitors_with_searxng(query)
        if urls:
            url_map[name] = urls[0]
            print(f"  {name} -> {urls[0]}")
        else:
            print(f"  {name} -> No URL found")
    return url_map

# ══════════════════════════════════════════════
# Industry Relevance Validator
# ══════════════════════════════════════════════
INDUSTRY_BLOCKLIST = {
    "archery", "hunting", "fishing", "bow", "arrow", "rifle", "ammunition",
    "garden", "farming", "tractor", "plumbing", "cooking", "recipe",
    "fashion", "clothing", "apparel", "footwear", "cosmetics",
}

def validate_relevance(scraped_text: str, industry: str, competitor_name: str) -> bool:
    """Checks if scraped content is relevant to the industry. Drops off-topic URLs."""
    text_lower = scraped_text[:2000].lower()
    blocklist_hits = sum(1 for word in INDUSTRY_BLOCKLIST if word in text_lower)
    tech_words = ["software", "app", "platform", "saas", "pricing", "plan", "feature",
                  "api", "integration", "dashboard", "team", "workspace", "project",
                  "collaborate", "download", "sign up", "free trial", "enterprise"]
    tech_hits = sum(1 for word in tech_words if word in text_lower)
    if blocklist_hits >= 3 and tech_hits < 2:
        print(f"    ⚠ RELEVANCE FAIL for {competitor_name}: {blocklist_hits} blocklist hits — DROPPING")
        return False
    return True

# ══════════════════════════════════════════════
# Phase 2-4: Extractors (Wayback + Live + Reddit)
# ══════════════════════════════════════════════
async def run_extractor_agent(url_map: Dict[str, str], industry: str) -> tuple:
    """Extracts historical + live data per competitor. Validates relevance. Adds Reddit Sentiment."""
    if not url_map:
        return "No competitor URLs found.", []

    extracted_data = []
    live_urls = []
    dropped_names = []

    first_competitor = True
    for name, url in url_map.items():
        # Cooldown between competitors to avoid CDX rate limiting
        if not first_competitor:
            print(f"  [cooldown] 2s pause between CDX requests...")
            await asyncio.sleep(2)
        first_competitor = False

        print(f"  [{name}] Scraping: {url}")

        # Phase 2: Wayback (always returns dict, never None, retries internally)
        wb = await get_wayback_snapshot(url)
        wb_status = wb.get("wayback_status", "unknown")
        wb_text = wb.get("extracted_text", "")
        wb_date = wb.get("snapshot_date", "N/A")
        wb_url = wb.get("wayback_url", "N/A")

        # Phase 3: Live Scrape
        live_md = await extract_markdown_with_firecrawl(url)
        print(f"    Firecrawl: {len(live_md)} chars")

        # Phase 4 (Async Sentiment): Reddit 
        print(f"    Reddit: Fetching top complaints for {name}...")
        reddit_text = await asyncio.to_thread(get_reddit_sentiment_sync, name)

        # Relevance validation
        if not validate_relevance(live_md, industry, name):
            dropped_names.append(name)
            extracted_data.append(
                f"--- COMPETITOR: {name} ---\n"
                f"[DROPPED: Content not relevant to {industry}. URL pointed to wrong company.]"
            )
            continue

        # Label data clearly for the LLM 
        extracted_data.append(
            f"--- COMPETITOR: {name} ---\n"
            f"[LIVE SCRAPE from {url}]\n{live_md[:6000]}\n\n"
            f"[WAYBACK 6-MONTH SCRAPE from {wb_url} | Date: {wb_date} | Status: {wb_status}]\n{wb_text[:5000]}\n\n"
            f"[REDDIT SENTIMENT]\n{reddit_text}"
        )
        live_urls.append(url)

    if dropped_names:
        print(f"  ⚠ Dropped {len(dropped_names)} irrelevant URLs: {dropped_names}")

    combined = "\n\n".join(extracted_data)
    print(f"  Total extraction payload: {len(combined)} chars")
    return combined, live_urls

# ══════════════════════════════════════════════
# Archiver 
# ══════════════════════════════════════════════
async def run_archiver_agent(urls: List[str]) -> Dict[str, str]:
    """Pushes live competitor pages into the Wayback Machine."""
    archive_map = {}
    for url in urls:
        print(f"  Archiving: {url}")
        link = await archive_to_wayback(url)
        if link:
            archive_map[url] = link
    return archive_map

# ══════════════════════════════════════════════
# Phase 5: Elite Market Intelligence Synthesizer
# ══════════════════════════════════════════════
async def run_normalization_agent(
    raw_data: str,
    user_prompt: str,
    understanding: Dict[str, str],
    archive_map: Dict[str, str]
) -> dict:
    """LLM ONLY does Phase 5: Synthesis & Analysis. Outputs JSON."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    industry = understanding.get("industry", "unknown")

    competitors_known = ", ".join([
        understanding.get(f"competitor_{i}", "")
        for i in range(1, 6)
        if understanding.get(f"competitor_{i}")
    ])

    archive_proof = "\n".join(
        [f"  {url} -> {link}" for url, link in archive_map.items()]
    ) if archive_map else "No archives created."

    system_prompt = f"""[SYSTEM INSTRUCTIONS]
You are the elite Market Intelligence Synthesizer for the MarketLens BI Engine.
Your ONLY job is to take raw, pre-scraped data from multiple sources (Live Website Markdown, Historical Wayback Machine text, and Reddit API sentiment) and synthesize it into a strict analytical payload.

The user is building a startup in the "{industry}" space.
Known competitors: {competitors_known}.

You operate under three absolute directives:
1. JSON ONLY. You must output strictly in valid flat JSON format.
2. NO HALLUCINATION. If data is missing from the provided text, output "Insufficient Data". Do not guess.
3. NO FLUFF. Never output generic business advice like "focus on unique value" or "leverage strengths." Market insights must contain specific mathematical pricing gaps, feature vacuums, or exact wording overlaps.

[DATA INPUT FORMAT]
You will receive data formatted as:
--- COMPETITOR: {{Name}} ---
[LIVE SCRAPE]...
[WAYBACK 6-MONTH SCRAPE]...
[REDDIT SENTIMENT]...

[REQUIRED JSON OUTPUT SCHEMA]
Structure your output exactly like this example JSON. 
{{
  "analysis_metadata_target_company": "UserStartupName",
  "analysis_metadata_target_industry": "Productivity Software",
  "analysis_metadata_scraped_at": "2026-03-25T16:00:00Z",
  "analysis_metadata_competitors_analyzed": 2,
  
  "competitor_1_name": "Monday.com",
  "competitor_1_current_base_price": "$12 per seat/month",
  "competitor_1_historical_base_price": "$9 per seat/month (Sep 2025)",
  "competitor_1_historical_price_delta": "+$3/seat (+33% increase in 6 months)",
  "competitor_1_top_reddit_complaint": "Offline mode sync failures delete work",
  "competitor_1_hero_tagline": "A platform built for a new way of working",
  "competitor_1_wayback_status": "OK",
  "competitor_1_archive_proof_url": "https://web.archive.org/web/20260325/monday.com/pricing",

  "competitor_2_name": "ClickUp",
  "competitor_2_current_base_price": "$7 per member/month",
  "competitor_2_historical_base_price": "Insufficient Data",
  "competitor_2_historical_price_delta": "Insufficient Data",
  "competitor_2_top_reddit_complaint": "Feature overload causes massive UI latency",
  "competitor_2_hero_tagline": "One app to replace them all",
  "competitor_2_wayback_status": "CDX API request timed out after 15s",
  "competitor_2_archive_proof_url": "https://web.archive.org/web/20260325/clickup.com/pricing",

  "market_intelligence_pricing_vacuum": "No competitor offers a flat-rate team plan. All use per-seat pricing ($7-$12/seat). A $49/mo unlimited-seats plan would undercut the entire market for teams of 5+.",
  "market_intelligence_feature_whitespace": "All 5 competitors lack native offline-first architecture. This is the #1 Reddit complaint visible across scraped marketing copy.",
  "market_intelligence_overused_messaging_cluster": "4 out of 5 competitors use variations of 'all-in-one' or 'one tool' in their hero tagline. Differentiate by avoiding this phrase entirely.",
  "market_intelligence_pricing_trend": "Monday.com raised Basic from $9 to $12/seat (+33%) in 6 months. Market is trending toward price increases — opportunity to lock in lower pricing as a growth lever."
}}

[EXECUTION INSTRUCTIONS]
1. Read the provided raw data for each competitor.
2. Extract current_base_price from the LIVE SCRAPE data (lowest paid tier).
3. Extract historical_base_price from the WAYBACK SCRAPE data.
4. Calculate historical_price_delta by comparing live vs historical (include % change).
5. Extract the top structural complaint from the REDDIT SENTIMENT section.
6. Calculate the market_intelligence section using facts from the text.
7. Output ONLY valid, parsable JSON. No markdown backticks, no explanatory text, no nested objects.
8. If a competitor was DROPPED, still include it but write "DROPPED: URL pointed to wrong company" for all fields."""

    llm_prompt = (
        f"Client Startup: {user_prompt}\n\n"
        f"Archive Proof URLs:\n{archive_proof}\n\n"
        f"[USER RAW DATA PAYLOAD BEGINS BELOW]\n{raw_data}"
    )

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": llm_prompt}
        ],
        "temperature": 0.1,
    }

    print("  Cooling down 15s for Groq rate limit...")
    await asyncio.sleep(15)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers, json=payload, timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                raw_output = data["choices"][0]["message"]["content"].strip()
                return json.loads(raw_output)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                wait = (attempt + 1) * 10
                print(f"  Rate limited. Retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                raise ValueError(f"LLM HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            if isinstance(e, json.JSONDecodeError):
                raise ValueError(f"LLM did not return valid JSON: {str(e)}")
            raise ValueError(f"LLM Error: {str(e)}")

# ══════════════════════════════════════════════
# Phase 6: Storage (File Export)
# ══════════════════════════════════════════════
async def save_to_local_file(data: Dict[str, Any]):
    os.makedirs("data_exports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    company = data.get("analysis_metadata_target_company", "analysis")
    company = company.replace(" ", "_").lower() if isinstance(company, str) else "analysis"

    json_path = os.path.join("data_exports", f"{company}_{ts}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n[+] Saved JSON  -> {json_path}")

# ══════════════════════════════════════════════
# API Endpoint
# ══════════════════════════════════════════════
@app.post("/api/v1/analyze", response_class=JSONResponse)
async def analyze_competitor(request: AnalyzeRequest):
    try:
        print("\n" + "=" * 60)
        print("MarketLens BI Engine v3.1 — Pipeline Started")
        print("=" * 60)

        # Phase 0: Understand the startup
        print("\n[Phase 0] Understanding startup description...")
        understanding = await run_understanding_agent(request.user_prompt)
        industry = understanding.get("industry", "unknown")
        for i in range(1, 6):
            c = understanding.get(f"competitor_{i}")
            if c:
                print(f"  Competitor {i}: {c}")
        print(f"  Industry: {industry}")

        # Phase 1: Discovery (industry-constrained)
        print("\n[Phase 1] Discovering competitor URLs (industry-constrained)...")
        url_map = await run_discovery_agent(understanding)
        print(f"  Found {len(url_map)} competitor URLs")

        # Phase 2-4: Extraction (Wayback, Firecrawl, Reddit)
        print("\n[Phase 2-4] Extracting historical, live, and sentiment data...")
        raw_text, live_urls = await run_extractor_agent(url_map, industry)

        # Archiver (parallel with Groq cooldown)
        print("\n[Archiver] Pushing live snapshots to Wayback Machine...")
        archive_task = asyncio.create_task(run_archiver_agent(live_urls))

        # Phase 5: Elite Synthesis
        print("\n[Phase 5] Running Elite Market Intelligence Synthesizer...")
        archive_map = await archive_task
        print(f"  Archived {len(archive_map)} pages to Wayback Machine")
        
        data = await run_normalization_agent(raw_text, request.user_prompt, understanding, archive_map)

        # Phase 6: Decode and save
        print("\n[Phase 6] Saving Output...")
        await save_to_local_file(data)

        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)

        return {
            "parsed_json": data,
            "archive_proof": archive_map
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

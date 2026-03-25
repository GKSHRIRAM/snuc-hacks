import os
import json
from dotenv import load_dotenv

load_dotenv()

import asyncio
import httpx
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

# Allow the local frontend (file:// or localhost) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    user_prompt: str

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared LLM helper — Groq (free, 14,400 req/day)
# Uses llama-3.1-8b-instant: smallest/fastest model
# to stay within 6000 tokens/min free-tier limit.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROQ_MODEL   = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

async def _groq_chat(messages: list, max_tokens: int = 1024, max_retries: int = 4) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload  = {"model": GROQ_MODEL, "messages": messages,
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

async def run_understanding_agent(prompt: str) -> Dict[str, str]:
    """Phase 0: Identifies industry and lists 5 real competitors."""
    system_prompt = (
        "You are a market research analyst. Identify the software industry and name the top 5 REAL "
        "tech/software competitors for the user's startup. Include incumbents and challengers. "
        "ONLY name companies in the SAME industry.\n"
        "Respond in this EXACT format (no extra text):\n"
        "industry: <industry name>\n"
        "competitor_1: <full company name>\n"
        "competitor_2: <full company name>\n"
        "competitor_3: <full company name>\n"
        "competitor_4: <full company name>\n"
        "competitor_5: <full company name>"
    )

    raw = await _groq_chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ], max_tokens=256)

    result = {}
    for line in raw.split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip()] = v.strip()
    return result if result else {"industry": "unknown"}

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

def _truncate(s: str, n: int) -> str:
    """Return first n characters of s. Avoids Pyre2 slice-overload false positives."""
    return s if len(s) <= n else s[0:n]  # type: ignore[index]

def validate_relevance(scraped_text: str, industry: str, competitor_name: str) -> bool:
    """Checks if scraped content is relevant to the industry. Drops off-topic URLs."""
    text_lower = _truncate(str(scraped_text), 2000).lower()
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
        live_md: str = str(await extract_markdown_with_firecrawl(url))
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
            f"[LIVE SCRAPE from {url}]\n{_truncate(live_md, 6000)}\n\n"
            f"[WAYBACK 6-MONTH SCRAPE from {wb_url} | Date: {wb_date} | Status: {wb_status}]\n{_truncate(str(wb_text), 5000)}\n\n"
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
    """Phase 5: Synthesis & Analysis. Calls Gemini, outputs flat JSON."""
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
        f"[RAW DATA PAYLOAD]\n{raw_data}"
    )

    print("  Calling Groq for synthesis...")
    try:
        raw_output = await _groq_chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": llm_prompt}
        ], max_tokens=2048)

        # Strip potential markdown fences
        content = raw_output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Groq did not return valid JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Gemini LLM Error: {str(e)}")

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

class InsightsRequest(BaseModel):
    export_path: str

@app.post("/api/v1/insights")
async def generate_insights(request: InsightsRequest, background_tasks: BackgroundTasks):
    if not os.path.exists(request.export_path):
        raise HTTPException(status_code=400, detail="Export file not found.")

    try:
        from differ import diff_from_file
        diff = diff_from_file(request.export_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to normalise or diff: {str(e)}")

    try:
        from insight_engine import get_insights, save_insights
        insights = await get_insights(diff)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Perplexity API failed: {str(e)}")

    def background_save(insights_data, diff_data):
        save_insights(insights_data, diff_data)
        os.makedirs("data_exports/diffs", exist_ok=True)
        ts = diff_data.export_timestamp.strftime("%Y%m%d_%H%M%S")
        company = diff_data.startup_query.replace(" ", "_").lower() if diff_data.startup_query else "analysis"
        diff_path = os.path.join("data_exports/diffs", f"{company}_{ts}_diff.json")
        try:
            with open(diff_path, "w", encoding="utf-8") as f:
                f.write(diff_data.model_dump_json(indent=2))
        except Exception as e:
            print(f"Failed to save diff locally: {e}")

    background_tasks.add_task(background_save, insights, diff)

    diff_list_clean = []
    for cd in diff.competitor_diffs:
        diff_list_clean.append({
            "competitor_name": cd.competitor_name,
            "period": {
                "from": cd.historical_date.isoformat() if cd.historical_date else None,
                "to": cd.live_date.isoformat() if cd.live_date else None
            },
            "pricing_changes": [p.model_dump() for p in cd.pricing_changes],
            "features_added": cd.features_added,
            "features_removed": cd.features_removed,
            "sentiment_score_delta": cd.sentiment_score_delta,
            "new_complaints": cd.new_complaints,
            "programmatic_summary": cd.programmatic_summary
        })

    return {
        "meta": {
            "startup_query": diff.startup_query,
            "industry": diff.industry,
            "export_timestamp": diff.export_timestamp.isoformat(),
            "competitors_analysed": len(diff.competitor_diffs)
        },
        "diffs": diff_list_clean,
        "insights": insights
    }

@app.post("/api/v1/analyze-and-insights")
async def analyze_and_insights(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    analysis_res = await analyze_competitor(request)
    
    # Locate the most recently saved raw export file (only in root of data_exports, not subdirs)
    exports_dir = "data_exports"
    newest_file = None
    if os.path.exists(exports_dir):
        files = [
            os.path.join(exports_dir, f)
            for f in os.listdir(exports_dir)
            if f.endswith(".json") and os.path.isfile(os.path.join(exports_dir, f))
        ]
        if files:
            newest_file = max(files, key=os.path.getmtime)
            
    if not newest_file:
         raise HTTPException(status_code=500, detail="Failed to locate the saved export file after analysis.")
         
    insights_req = InsightsRequest(export_path=newest_file)
    insights_res = await generate_insights(insights_req, background_tasks)
    
    return {
        "analysis_raw": analysis_res,
        "insights_pipeline": insights_res
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

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

        # Truncate each source — sufficient for pricing + features + sentiment analysis
        live_snippet   = _truncate(live_md, 2500)   # enough for pricing table, features, tagline
        wb_snippet     = _truncate(str(wb_text), 1800)  # 12-mo historical snapshot
        reddit_snippet = _truncate(str(reddit_text), 700)  # top complaints + praise

        extracted_data.append(
            f"--- COMPETITOR: {name} ---\n"
            f"[LIVE SCRAPE | {url}]\n{live_snippet}\n\n"
            f"[WAYBACK SNAPSHOT | Date: {wb_date} | Status: {wb_status}]\n{wb_snippet}\n\n"
            f"[REDDIT COMMUNITY SENTIMENT]\n{reddit_snippet}"
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

    system_prompt = (
        f"You are a senior competitive intelligence analyst. Industry: {industry}. "
        f"Competitors analysed: {competitors_known}.\n\n"
        "Produce a SINGLE valid JSON object (no markdown fences, no extra text) with this EXACT structure:\n"
        "{\n"
        '  "meta": {"startup_industry": str, "scraped_at": ISO8601, "competitors_count": int, "snapshot_gap_months": 12},\n'
        '  "competitors": {\n'
        '    "<CompetitorName>": {\n'
        '      "tagline": str,\n'
        '      "target_segment": str,\n'
        '      "positioning": "budget|mid-market|premium|enterprise",\n'
        '      "pricing_model": "per-seat|flat-rate|usage-based|freemium|free",\n'
        '      "current_pricing": [{"tier": str, "price_usd": float_or_null, "billing": "monthly|annual|one-time"}],\n'
        '      "historical_pricing": [{"tier": str, "price_usd": float_or_null}],\n'
        '      "pricing_delta_pct": float_or_null,\n'
        '      "key_features": [str],\n'
        '      "recent_additions": [str],\n'
        '      "recent_removals": [str],\n'
        '      "competitive_moat": str,\n'
        '      "messaging_tone": "technical|consumer|enterprise|developer",\n'
        '      "top_complaints": [str],\n'
        '      "top_praise": [str],\n'
        '      "sentiment_score": float_between_neg1_and_1,\n'
        '      "wayback_status": str\n'
        '    }\n'
        '  },\n'
        '  "market_analysis": {\n'
        '    "common_features": [str],\n'
        '    "feature_gaps": [str],\n'
        '    "pricing_range": {"min_usd": float, "max_usd": float, "avg_usd": float},\n'
        '    "pricing_trend": str,\n'
        '    "pricing_vacuum": str,\n'
        '    "overused_messaging": [str],\n'
        '    "market_positioning_map": [{"name": str, "price_score": 0-10, "feature_richness": 0-10, "sentiment_score": -1_to_1}],\n'
        '    "entry_opportunities": [str],\n'
        '    "differentiation_angles": [str]\n'
        '  }\n'
        "}\n\n"
        "Rules:\n"
        "1. Extract all pricing from the LIVE data. Extract historical pricing from WAYBACK data.\n"
        "2. pricing_delta_pct = ((current - historical) / historical * 100) if both exist, else null.\n"
        "3. Use Reddit data for top_complaints and top_praise.\n"
        "4. market_positioning_map: price_score 1=cheapest/free .. 10=most expensive; feature_richness 1=bare .. 10=full-featured.\n"
        "5. If data is missing, use null for numbers and 'Insufficient Data' for strings.\n"
        "6. Output valid JSON only — no markdown, no explanation text."
    )

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
        ], max_tokens=3000)

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

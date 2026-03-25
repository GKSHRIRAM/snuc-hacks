import os
import json
import uuid
from dotenv import load_dotenv

load_dotenv()

import asyncio
import httpx
from datetime import datetime
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tools.searxng import search_competitors_with_searxng
from tools.wayback import get_wayback_snapshot
from tools.firecrawl_extractor import extract_markdown_with_firecrawl
from tools.wayback_archiver import archive_to_wayback
from reviews.engine.review_engine import ReviewEngine

app = FastAPI(
    title="MarketLens BI Engine",
    description="Multi-source market intelligence pipeline with Review Sentinel.",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# In-Memory Job Tracker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
jobs: Dict[str, Dict[str, Any]] = {}

class AnalyzeRequest(BaseModel):
    user_prompt: str

from tools.llm_client import llm_chat

def _truncate(s: str, n: int) -> str:
    """Return the first n characters of a string."""
    if len(s) <= n:
        return s
    import itertools
    return "".join(itertools.islice(s, n))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: Understanding Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def run_understanding_agent(prompt: str) -> Dict[str, str]:
    """Identifies industry and lists 5 real competitors."""
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
    raw = await llm_chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ], json_mode=False)

    result = {}
    for line in raw.split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip()] = v.strip()
    return result if result else {"industry": "unknown"}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Discovery Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def run_discovery_agent(understanding: Dict[str, str]) -> Dict[str, str]:
    """Discovers actual URLs via DuckDuckGo HTML scraping."""
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Relevance Validator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INDUSTRY_BLOCKLIST = {
    "archery", "hunting", "fishing", "bow", "arrow", "rifle", "ammunition",
    "garden", "farming", "tractor", "plumbing", "cooking", "recipe",
    "fashion", "clothing", "apparel", "footwear", "cosmetics",
}

def validate_relevance(scraped_text: str, industry: str, competitor_name: str) -> bool:
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Review Sentinel (Groq-powered, no external API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def get_customer_reviews(competitor_name: str) -> dict:
    """Extracts review sentiment via Groq LLM knowledge."""
    print(f"    [Review Sentinel] Analyzing reviews for {competitor_name}...")
    prompt = (
        f"Based on your knowledge of the software product '{competitor_name}', provide:\n"
        "1. What users typically praise (3 specific points)\n"
        "2. What users typically complain about (3 specific points)\n"
        "3. Common feature requests from users (2 points)\n\n"
        "If you have no knowledge of this product, return empty lists."
    )
    system = 'Return ONLY valid JSON: {"positives":[], "negatives":[], "suggestions":[]}'
    try:
        raw = await llm_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ], json_mode=True)
        content = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(content)
    except Exception as e:
        print(f"    [Review Sentinel] Failed: {e}")
        return {"positives": [], "negatives": [], "suggestions": []}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reddit Sentiment (Groq-powered, no praw needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def get_reddit_sentiment(competitor_name: str) -> str:
    """Gets Reddit-style sentiment via Groq LLM knowledge."""
    print(f"    [Reddit Sentiment] Analyzing for {competitor_name}...")
    prompt = (
        f"What are the most common Reddit complaints and praise about '{competitor_name}' software? "
        "List 3-5 bullet points of real user complaints and 2-3 points of praise. "
        "If you don't know, say 'Insufficient Data'."
    )
    try:
        return await llm_chat([
            {"role": "system", "content": "You are a Reddit sentiment analyst. Be concise and specific."},
            {"role": "user", "content": prompt}
        ])
    except Exception as e:
        import itertools
        err_trunc = "".join(itertools.islice(str(e), 80))
        return f"Insufficient Data (Error: {err_trunc})"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trustpilot Sentiment (BeautifulSoup-powered)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
trustpilot_engine = ReviewEngine()

async def get_trustpilot_reviews(competitor_name: str, url: str) -> dict:
    """Extracts live Trustpilot data using the review engine."""
    print(f"    [Trustpilot] Scraping reviews for {competitor_name}...")
    try:
        # Extract domain from URL
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        if not domain:
            domain = url.split("/")[0] # fallback
        
        score, reviews = await asyncio.to_thread(trustpilot_engine.get_review_data, domain)
        return {
            "score": score,
            "reviews": reviews,
            "count": len(reviews)
        }
    except Exception as e:
        print(f"    [Trustpilot] Failed: {e}")
        return {"score": 0.0, "reviews": [], "count": 0}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2-4: Extraction Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def run_extractor_agent(url_map: Dict[str, str], industry: str, job_id: str = None) -> tuple:
    """Parallelized extraction — researches all competitors concurrently."""
    if not url_map:
        return ("No competitor URLs found.", {}), []

    competitor_payloads: Dict[str, Any] = {}
    live_urls: List[str] = []
    
    # Process up to 3 competitors concurrently to avoid system overload
    sem = asyncio.Semaphore(3)

    async def _process_competitor(name: str, url: str):
        async with sem:
            print(f"  [{name}] Extracting intelligence...")
            
            # Run Wayback + Live + Sentiment concurrently
            wb_task = get_wayback_snapshot(url)
            live_task = extract_markdown_with_firecrawl(url)
            reddit_task = get_reddit_sentiment(name)
            review_task = get_customer_reviews(name)
            tp_task = get_trustpilot_reviews(name, url)

            wb, live_md, reddit_text, reviews, tp_data = await asyncio.gather(
                wb_task, live_task, reddit_task, review_task, tp_task
            )

            live_md = str(live_md)
            wb_text = wb.get("extracted_text", "")
            wb_date = wb.get("snapshot_date", "N/A")
            wb_status = wb.get("wayback_status", "unknown")

            print(f"    [{name}] Done (Live: {len(live_md)} chars | TP Score: {tp_data['score']:.2f})")

            # Relevance check
            if not validate_relevance(live_md, industry, name):
                competitor_payloads[name] = {"dropped": True}
                return

            live_snippet   = _truncate(live_md, 2500)
            wb_snippet     = _truncate(str(wb_text), 1800)
            reddit_snippet = _truncate(str(reddit_text), 700)

            competitor_payloads[name] = {
                "raw_text": (
                    f"--- COMPETITOR: {name} ---\n"
                    f"[LIVE SCRAPE | {url}]\n{live_snippet}\n\n"
                    f"[WAYBACK SNAPSHOT | Date: {wb_date} | Status: {wb_status}]\n{wb_snippet}\n\n"
                    f"[REDDIT COMMUNITY SENTIMENT]\n{reddit_snippet}\n\n"
                    f"[TRUSTPILOT REVIEWS]\nScore: {tp_data['score']} | Count: {tp_data['count']}"
                ),
                "reviews": reviews,
                "trustpilot": tp_data,
                "url": url,
                "wayback_status": wb_status
            }
            live_urls.append(url)
            
            # Update granular progress
            if job_id and job_id in jobs:
                current_prog = jobs[job_id].get("progress", 35)
                # Increment by 35% / num_competitors
                increment = 35 / max(len(url_map), 1)
                jobs[job_id]["progress"] = min(70, current_prog + increment)

    # Launch all tasks
    tasks = [
        _process_competitor(name, url) for name, url in url_map.items()
    ]
    await asyncio.gather(*tasks)

    text_parts: List[str] = [
        str(p["raw_text"]) for p in competitor_payloads.values() if "raw_text" in p
    ]
    combined_text = "\n\n".join(text_parts)
    print(f"  Total extraction payload: {len(combined_text)} chars")
    return (combined_text, competitor_payloads), live_urls

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Archiver Agent (was missing — now restored)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def run_archiver_agent(urls: List[str]) -> Dict[str, str]:
    """Pushes live competitor pages into the Wayback Machine in parallel."""
    archive_map = {}
    capped = [u for i, u in enumerate(urls) if i < 5]  # limit to 5
    
    async def _archive(url: str):
        print(f"  Archiving: {url}")
        link = await archive_to_wayback(url)
        if link:
            archive_map[url] = link

    await asyncio.gather(*[_archive(u) for u in capped])
    return archive_map

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 5: Synthesis & Normalization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def run_normalization_agent(
    payload_data: tuple,
    user_prompt: str,
    understanding: Dict[str, str],
    archive_map: Dict[str, str]
) -> dict:
    """Synthesizes all intelligence into structured JSON via Groq."""
    raw_text, competitor_payloads = payload_data
    industry = understanding.get("industry", "unknown")
    competitors_known = ", ".join([
        name for name in competitor_payloads.keys() if not competitor_payloads[name].get("dropped")
    ])

    archive_proof = "\n".join(
        [f"  {url} -> {link}" for url, link in archive_map.items()]
    ) if archive_map else "No archives created."

    system_prompt = (
        f"You are a senior competitive intelligence analyst. Industry: {industry}. "
        f"Competitors analysed: {competitors_known}.\n\n"
        "Produce a SINGLE valid JSON object (no markdown fences, no explanation) with this structure:\n"
        "{\n"
        '  "meta": {"startup_industry": str, "scraped_at": ISO8601, "competitors_count": int, "snapshot_gap_months": 12},\n'
        '  "competitors": {\n'
        '    "<Name>": {\n'
        '      "tagline": str, "target_segment": str,\n'
        '      "positioning": "budget|mid-market|premium|enterprise",\n'
        '      "pricing_model": "per-seat|flat-rate|usage-based|freemium|free",\n'
        '      "current_pricing": [{"tier": str, "price_usd": float_or_null, "billing": "monthly|annual"}],\n'
        '      "historical_pricing": [{"tier": str, "price_usd": float_or_null}],\n'
        '      "pricing_delta_pct": float_or_null,\n'
        '      "key_features": [str], "competitive_moat": str,\n'
        '      "top_complaints": [str], "top_praise": [str],\n'
        '      "sentiment_score": float_neg1_to_1,\n'
        '      "review_sentiment": {"positives": [str], "negatives": [str], "suggestions": [str]},\n'
        '      "wayback_status": str\n'
        '    }\n'
        '  },\n'
        '  "market_analysis": {\n'
        '    "common_features": [str], "feature_gaps": [str],\n'
        '    "pricing_range": {"min_usd": float, "max_usd": float, "avg_usd": float},\n'
        '    "pricing_trend": str, "pricing_vacuum": str,\n'
        '    "overused_messaging": [str],\n'
        '    "entry_opportunities": [str], "differentiation_angles": [str]\n'
        '  }\n'
        "}\n\n"
        "Rules: Extract pricing from LIVE data. Historical from WAYBACK. "
        "Prioritize Trustpilot data for 'top_complaints', 'top_praise', and 'sentiment_score'. "
        "Use General Intelligence reviews as a secondary source. "
        "If data missing, use null / 'Insufficient Data'. Output ONLY valid JSON."
    )

    # Build review context
    review_parts: List[str] = ["\n[CUSTOMER REVIEW DATA]"]
    for name, p in competitor_payloads.items():
        if "reviews" in p:
            review_parts.append(f"--- {name} (General Intelligence) ---\n{json.dumps(p['reviews'], indent=2)}")
        if "trustpilot" in p:
            tp = p["trustpilot"]
            # To avoid context bloat, we take the top 10 reviews
            tp_reviews = tp.get("reviews", [])[:10]
            review_parts.append(f"--- {name} (Trustpilot Scrape) ---\nScore: {tp.get('score')} | Reviews:\n{json.dumps(tp_reviews, indent=2)}")
    
    review_block: str = "\n".join(review_parts)

    llm_prompt = (
        f"Client Startup: {user_prompt}\n"
        f"Archive Proof URLs:\n{archive_proof}\n\n"
        f"[RAW DATA]\n{raw_text}\n"
        f"{review_block}"
    )

    print("  [Phase 5] Calling Local LLM for synthesis...")

    try:
        raw_output = await llm_chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": llm_prompt}
        ], json_mode=True)

        content = raw_output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        final_json = json.loads(content)

        # Ensure review data is always present
        if "competitors" in final_json:
            for name, p in competitor_payloads.items():
                if name in final_json["competitors"]:
                    if "reviews" in p:
                        final_json["competitors"][name]["review_sentiment"] = p["reviews"]
                    if "trustpilot" in p:
                        final_json["competitors"][name]["trustpilot_data"] = p["trustpilot"]
        
        return final_json
    except json.JSONDecodeError as e:
        import itertools
        err_content = "".join(itertools.islice(content, 200))
        raise ValueError(f"Local LLM did not return valid JSON: {str(e)}\nOutput was: {err_content}")
    except Exception as e:
        raise ValueError(f"Synthesis Error: {str(e)}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Storage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def save_to_local_file(data: Dict[str, Any]) -> str:
    os.makedirs("data_exports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join("data_exports", f"marketlens_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\n[+] Saved JSON -> {json_path}")
    return json_path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Background Pipeline Orchestrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def execute_pipeline(job_id: str, user_prompt: str):
    """Full async pipeline — runs in background, updates job status."""
    try:
        jobs[job_id]["status"] = "UNDERSTANDING"
        jobs[job_id]["progress"] = 10
        print(f"\n{'='*60}\nMarketLens v4.0 — Pipeline Started (Job: {job_id})\n{'='*60}")

        # Phase 0
        print("\n[Phase 0] Understanding startup...")
        understanding = await run_understanding_agent(user_prompt)
        industry = understanding.get("industry", "unknown")
        print(f"  Industry: {industry}")
        for i in range(1, 6):
            c = understanding.get(f"competitor_{i}")
            if c: print(f"  Competitor {i}: {c}")

        # Phase 1
        jobs[job_id]["status"] = "DISCOVERING"
        jobs[job_id]["progress"] = 20
        print("\n[Phase 1] Discovering competitor URLs...")
        url_map = await run_discovery_agent(understanding)
        print(f"  Found {len(url_map)} URLs")

        # Phase 2-4
        jobs[job_id]["status"] = "EXTRACTING"
        jobs[job_id]["progress"] = 35
        print("\n[Phase 2-4] Extracting intelligence...")
        payload_data, live_urls = await run_extractor_agent(url_map, industry, job_id)

        # Archiver (parallel)
        jobs[job_id]["status"] = "ARCHIVING"
        jobs[job_id]["progress"] = 60
        print("\n[Archiver] Pushing to Wayback Machine...")
        archive_task = asyncio.create_task(run_archiver_agent(live_urls))

        # Phase 5
        jobs[job_id]["status"] = "SYNTHESIZING"
        jobs[job_id]["progress"] = 70
        print("\n[Phase 5] Synthesizing intelligence...")
        archive_map = await archive_task
        print(f"  Archived {len(archive_map)} pages")
        data = await run_normalization_agent(payload_data, user_prompt, understanding, archive_map)

        # Phase 6: Save
        jobs[job_id]["status"] = "SAVING"
        jobs[job_id]["progress"] = 90
        print("\n[Phase 6] Saving output...")
        export_path = await save_to_local_file(data)

        jobs[job_id]["status"] = "COMPLETED"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["result"] = data
        jobs[job_id]["export_path"] = export_path
        jobs[job_id]["archive_proof"] = archive_map
        print(f"\n{'='*60}\nPipeline Complete! (Job: {job_id})\n{'='*60}")

    except Exception as e:
        jobs[job_id]["status"] = "FAILED"
        jobs[job_id]["error"] = str(e)
        print(f"\n[PIPELINE ERROR] {e}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.post("/api/v1/analyze")
async def start_analysis(request: AnalyzeRequest):
    """Starts async pipeline, returns job ID for polling."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "QUEUED",
        "progress": 0,
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat()
    }
    asyncio.create_task(execute_pipeline(job_id, request.user_prompt))
    return {"job_id": job_id, "status": "QUEUED"}

@app.get("/api/v1/status/{job_id}")
async def get_status(job_id: str):
    """Poll this endpoint to check job progress."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"]
    }
    if job["status"] == "COMPLETED":
        response["result"] = job["result"]
        response["export_path"] = job.get("export_path")
        response["archive_proof"] = job.get("archive_proof", {})
    elif job["status"] == "FAILED":
        response["error"] = job["error"]
    return response

@app.post("/api/v1/analyze-sync")
async def analyze_sync(request: AnalyzeRequest):
    """Synchronous endpoint — blocks until pipeline completes (for direct API users)."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "QUEUED", "progress": 0,
        "result": None, "error": None,
        "created_at": datetime.now().isoformat()
    }
    await execute_pipeline(job_id, request.user_prompt)
    job = jobs[job_id]
    if job["status"] == "FAILED":
        raise HTTPException(status_code=500, detail=job["error"])
    return {"parsed_json": job["result"], "archive_proof": job.get("archive_proof", {})}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# New Dashboard Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class InsightsRequest(BaseModel):
    export_path: str

@app.post("/api/v1/insights")
async def generate_insights_endpoint(request: InsightsRequest):
    """Generates analytical insights from a saved export file."""
    if not os.path.exists(request.export_path):
        raise HTTPException(status_code=400, detail="Export file not found.")

    try:
        from differ import diff_from_file
        diff = diff_from_file(request.export_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to normalise or diff: {str(e)}")

    try:
        from insight_engine import get_insights
        insights = await get_insights(diff)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM Insights failed: {str(e)}")

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
async def analyze_and_insights(request: AnalyzeRequest):
    """Synchronous combined endpoint for the dashboard."""
    # Run analysis
    analysis_res = await analyze_sync(request)
    
    # Locate the newest export file
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
         raise HTTPException(status_code=500, detail="Failed to locate export file.")

    # Generate insights
    insights_req = InsightsRequest(export_path=newest_file)
    insights_res = await generate_insights_endpoint(insights_req)

    return {"analysis_raw": analysis_res, "insights_pipeline": insights_res}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

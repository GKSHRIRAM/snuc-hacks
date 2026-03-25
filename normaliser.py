import os
import json
import logging
import re
from datetime import datetime
from typing import Literal, List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════
# Pydantic v2 Schemas
# ══════════════════════════════════════════════

class CompetitorSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    competitor_name: str
    snapshot_date: datetime
    snapshot_type: Literal["live", "historical"]
    pricing_tiers: List[Dict[str, Any]]
    headline_features: List[str]
    reddit_sentiment: Dict[str, Any]
    raw_source_url: str

class NormalisedExport(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    startup_query: str
    industry: str
    generated_at: datetime
    snapshots: List[CompetitorSnapshot]

# ══════════════════════════════════════════════
# Utility Functions
# ══════════════════════════════════════════════

def extract_datetime_from_filename(filename: str) -> datetime:
    """Parses datetime from format <startup_name>_<YYYYMMDD_HHMMSS>.json"""
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    match = re.search(r'(\d{8}_\d{6})$', name_without_ext)
    if match:
        date_time_str = match.group(1)
        try:
            return datetime.strptime(date_time_str, "%Y%m%d_%H%M%S")
        except ValueError as e:
            logger.warning(f"Failed to parse datetime from filename '{filename}': {e}")
            
    return datetime.now()

def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "Insufficient Data":
        return None
    try:
        match = re.search(r'\d+(\.\d+)?', str(val))
        if match:
            return float(match.group())
    except Exception:
        pass
    return None

# ══════════════════════════════════════════════
# Normalise Function
# ══════════════════════════════════════════════

def normalise(raw_export_path: str) -> NormalisedExport:
    try:
        with open(raw_export_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load JSON from {raw_export_path}: {e}")
        raw_data = {}

    snapshot_date = extract_datetime_from_filename(raw_export_path)
    startup_query = str(raw_data.get("analysis_metadata_target_company", "Unknown"))
    industry = str(raw_data.get("analysis_metadata_target_industry", "Unknown"))
    
    generated_at_str = raw_data.get("analysis_metadata_scraped_at")
    generated_at = snapshot_date
    if generated_at_str:
        try:
            generated_at = datetime.fromisoformat(generated_at_str.replace("Z", "+00:00"))
        except Exception:
            pass

    snapshots = []
    
    for i in range(1, 6): # Parse competitor_1 to competitor_5
        comp_prefix = f"competitor_{i}_"
        comp_name_key = f"{comp_prefix}name"
        
        if comp_name_key not in raw_data:
            continue
            
        name = raw_data.get(comp_name_key)
        if not name or "DROPPED:" in str(name) or name == "Insufficient Data":
            continue
            
        # Extract features/taglines / Reddit Sentiment (Live)
        headline_features = []
        tagline = raw_data.get(f"{comp_prefix}hero_tagline")
        if tagline and tagline != "Insufficient Data":
            headline_features.append(str(tagline))
            
        reddit_sentiment = {
            "overall_score": 0.0,
            "top_complaints": [],
            "top_praises": []
        }
        
        complaint = raw_data.get(f"{comp_prefix}top_reddit_complaint")
        if complaint and complaint != "Insufficient Data":
            reddit_sentiment["top_complaints"].append(str(complaint))
            
        raw_source_url = raw_data.get(f"{comp_prefix}archive_proof_url", "")
        if raw_source_url == "Insufficient Data":
            raw_source_url = ""
            
        # Process live pricing tier
        live_price_str = raw_data.get(f"{comp_prefix}current_base_price")
        live_tiers = []
        if live_price_str and live_price_str != "Insufficient Data":
            live_tiers.append({
                "tier_name": "Base Plan",
                "price_usd": _safe_float(live_price_str),
                "billing_cycle": "monthly" if "month" in str(live_price_str).lower() else None,
                "features": []
            })
            
        # Process historical pricing tier
        hist_price_str = raw_data.get(f"{comp_prefix}historical_base_price")
        hist_tiers = []
        if hist_price_str and hist_price_str != "Insufficient Data":
            hist_tiers.append({
                "tier_name": "Base Plan",
                "price_usd": _safe_float(hist_price_str),
                "billing_cycle": "monthly" if "month" in str(hist_price_str).lower() else None,
                "features": []
            })
            
        try:
            # Create Live Snapshot
            snapshots.append(CompetitorSnapshot(
                competitor_name=name,
                snapshot_date=snapshot_date,
                snapshot_type="live",
                pricing_tiers=live_tiers,
                headline_features=headline_features.copy(),
                reddit_sentiment=reddit_sentiment.copy(),
                raw_source_url=raw_source_url
            ))
            
            # Create Historical Snapshot
            snapshots.append(CompetitorSnapshot(
                competitor_name=name,
                snapshot_date=snapshot_date, # Fallback, as exact historical date is embedded in string
                snapshot_type="historical",
                pricing_tiers=hist_tiers,
                headline_features=[], # pipeline doesn't extract historical features separately
                reddit_sentiment={"overall_score": 0.0, "top_complaints": [], "top_praises": []},
                raw_source_url=raw_source_url
            ))
        except Exception as e:
            logger.warning(f"Failed to validate competitor '{name}' fields. Skipping. Error: {e}")

    return NormalisedExport(
        startup_query=str(startup_query),
        industry=str(industry),
        generated_at=generated_at,
        snapshots=snapshots
    )

def save_normalised(snapshot: NormalisedExport, out_dir: str = "data_exports/normalised/"):
    try:
        os.makedirs(out_dir, exist_ok=True)
        ts = snapshot.generated_at.strftime("%Y%m%d_%H%M%S")
        company = snapshot.startup_query.replace(" ", "_").lower() if snapshot.startup_query else "analysis"
        filename = f"{company}_{ts}_normalised.json"
        out_path = os.path.join(out_dir, filename)
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(snapshot.model_dump_json(indent=2))
        print(f"[+] Saved Normalised JSON -> {out_path}")
    except Exception as e:
        logger.warning(f"Failed to save normalised data to {out_dir}: {e}")

from datetime import datetime
from typing import Literal, List, Optional
from pydantic import BaseModel, ConfigDict
from collections import defaultdict

# Must import from normaliser
from normaliser import NormalisedExport, normalise, CompetitorSnapshot

class PricingChange(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tier_name: str
    change_type: Literal["price_increase", "price_decrease", "tier_added", "tier_removed", "unchanged"]
    historical_price: Optional[float]
    live_price: Optional[float]
    delta_usd: Optional[float]

class CompetitorDiff(BaseModel):
    model_config = ConfigDict(extra="ignore")
    competitor_name: str
    historical_date: Optional[datetime]
    live_date: Optional[datetime]
    pricing_changes: List[PricingChange]
    features_added: List[str]
    features_removed: List[str]
    sentiment_score_delta: Optional[float]
    new_complaints: List[str]
    resolved_complaints: List[str]
    programmatic_summary: str

class ExportDiff(BaseModel):
    model_config = ConfigDict(extra="ignore")
    startup_query: str
    industry: str
    export_timestamp: datetime
    competitor_diffs: List[CompetitorDiff]

def diff_export(export: NormalisedExport) -> ExportDiff:
    # Group CompetiitorSnapshots by name
    comp_map = defaultdict(lambda: {"live": None, "historical": None})
    
    for snap in export.snapshots:
        comp_map[snap.competitor_name][snap.snapshot_type] = snap
        
    diffs = []
    
    for name, snaps in comp_map.items():
        hist: Optional[CompetitorSnapshot] = snaps["historical"]
        live: Optional[CompetitorSnapshot] = snaps["live"]
        
        hist_date = hist.snapshot_date if hist else None
        live_date = live.snapshot_date if live else None
        
        # Features Diff
        hist_features = set(hist.headline_features) if hist else set()
        live_features = set(live.headline_features) if live else set()
        features_added = list(live_features - hist_features)
        features_removed = list(hist_features - live_features)
        
        # Sentiment Diff
        hist_score = hist.reddit_sentiment.get("overall_score", 0.0) if hist else 0.0
        live_score = live.reddit_sentiment.get("overall_score", 0.0) if live else 0.0
        sentiment_delta = live_score - hist_score if (hist and live) else None
        
        hist_complaints = set(hist.reddit_sentiment.get("top_complaints", [])) if hist else set()
        live_complaints = set(live.reddit_sentiment.get("top_complaints", [])) if live else set()
        new_complaints = list(live_complaints - hist_complaints)
        resolved_complaints = list(hist_complaints - live_complaints)
        
        # Pricing Diff
        pricing_changes = []
        hist_tiers = {t.get("tier_name", "Unknown").lower(): t for t in hist.pricing_tiers} if hist else {}
        live_tiers = {t.get("tier_name", "Unknown").lower(): t for t in live.pricing_tiers} if live else {}
        
        all_tier_keys = set(hist_tiers.keys()) | set(live_tiers.keys())
        
        tier_added_count = 0
        tier_removed_count = 0
        tier_changed_count = 0
        
        for t_key in all_tier_keys:
            h_tier = hist_tiers.get(t_key)
            l_tier = live_tiers.get(t_key)
            
            t_name = l_tier.get("tier_name") if l_tier else h_tier.get("tier_name")
            h_price = h_tier.get("price_usd") if h_tier else None
            l_price = l_tier.get("price_usd") if l_tier else None
            
            delta = None
            ctype: Literal["price_increase", "price_decrease", "tier_added", "tier_removed", "unchanged"] = "unchanged"
            
            if h_price is not None and l_price is not None:
                delta = l_price - h_price
                if delta > 0:
                    ctype = "price_increase"
                    tier_changed_count += 1
                elif delta < 0:
                    ctype = "price_decrease"
                    tier_changed_count += 1
                else:
                    ctype = "unchanged"
            elif h_price is None and l_price is not None:
                ctype = "tier_added" if hist else "unchanged" # Treat as unchanged if we lack historical context
                if ctype == "tier_added": tier_added_count += 1
            elif h_price is not None and l_price is None:
                ctype = "tier_removed" if live else "unchanged" # Treat as unchanged if we lack live context
                if ctype == "tier_removed": tier_removed_count += 1
            else:
                ctype = "unchanged"
                
            pricing_changes.append(PricingChange(
                tier_name=str(t_name),
                change_type=ctype,
                historical_price=h_price,
                live_price=l_price,
                delta_usd=delta
            ))
            
        # Programmatic Summary construction
        summary_parts = []
        
        if not hist and live:
            prog_summary = f"{name} only has live data (no historical snapshot)."
        elif hist and not live:
            prog_summary = f"{name} only has historical data (no live snapshot)."
        else:
            if tier_changed_count > 0:
                summary_parts.append(f"changed prices on {tier_changed_count} tier(s)")
            if tier_added_count > 0:
                summary_parts.append(f"added {tier_added_count} tier(s)")
            if tier_removed_count > 0:
                summary_parts.append(f"removed {tier_removed_count} tier(s)")
                
            if features_added:
                summary_parts.append(f"added {len(features_added)} features")
            if features_removed:
                summary_parts.append(f"removed {len(features_removed)} features")
                
            if sentiment_delta is not None and sentiment_delta != 0.0:
                verb = "worsened" if sentiment_delta < 0 else "improved"
                summary_parts.append(f"sentiment {verb} by {abs(sentiment_delta):.2f}")
                
            if not summary_parts:
                prog_summary = f"Since the historical snapshot, {name} made zero detectable changes."
            elif len(summary_parts) == 1:
                prog_summary = f"Since the historical snapshot, {name} {summary_parts[0]}."
            else:
                prog_summary = f"Since the historical snapshot, {name} " + ", ".join(summary_parts[:-1]) + f", and {summary_parts[-1]}."
                
        diffs.append(CompetitorDiff(
            competitor_name=name,
            historical_date=hist_date,
            live_date=live_date,
            pricing_changes=pricing_changes,
            features_added=features_added,
            features_removed=features_removed,
            sentiment_score_delta=sentiment_delta,
            new_complaints=new_complaints,
            resolved_complaints=resolved_complaints,
            programmatic_summary=prog_summary
        ))
        
    return ExportDiff(
        startup_query=export.startup_query,
        industry=export.industry,
        export_timestamp=export.generated_at,
        competitor_diffs=diffs
    )

def diff_from_file(raw_export_path: str) -> ExportDiff:
    """Takes a raw JSON export path, normalises it, and diffs the internal historical/live state."""
    export = normalise(raw_export_path)
    return diff_export(export)

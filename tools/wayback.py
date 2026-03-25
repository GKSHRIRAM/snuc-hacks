import re
import asyncio
import httpx
from typing import Dict
from datetime import datetime, timedelta

WAYBACK_USER_AGENT = "MarketLens-Research-Bot/1.0 (market intelligence tool; contact: marketlens@example.com)"

async def get_wayback_snapshot(url: str, max_retries: int = 3) -> Dict[str, str]:
    """
    Agent Tool: Ping the Wayback CDX API with exponential backoff retry.
    ALWAYS returns a dict — never None. On failure, includes wayback_status with the error reason.
    """
    target_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")

    cdx_url = (
        f"https://web.archive.org/cdx/search/cdx"
        f"?url={url}&output=json&limit=1&closest={target_date}&sort=closest"
    )

    headers = {"User-Agent": WAYBACK_USER_AGENT}

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                if attempt == 0:
                    print(f"    Wayback CDX query for {url} (target: {target_date})...")
                else:
                    print(f"    Wayback CDX retry {attempt + 1}/{max_retries} for {url}...")

                # Step 1: Query CDX API
                cdx_resp = await client.get(cdx_url, headers=headers, timeout=25.0)

                # Handle rate limiting
                if cdx_resp.status_code == 429:
                    wait = 2 ** attempt
                    print(f"    Wayback: HTTP 429 Rate Limited. Sleeping {wait}s...")
                    await asyncio.sleep(wait)
                    continue

                cdx_resp.raise_for_status()
                rows = cdx_resp.json()

                if len(rows) < 2:
                    print(f"    Wayback: No snapshots found for {url}")
                    return {
                        "snapshot_date": "N/A",
                        "wayback_url": "N/A",
                        "wayback_status": "No archived snapshots exist for this URL",
                        "extracted_text": ""
                    }

                header = rows[0]
                data = rows[1]
                row_dict = dict(zip(header, data))

                timestamp = row_dict.get("timestamp", "")
                snapshot_url = f"https://web.archive.org/web/{timestamp}/{url}"

                try:
                    snapshot_date = datetime.strptime(timestamp[:8], "%Y%m%d").strftime("%Y-%m-%d")
                except:
                    snapshot_date = timestamp[:8]

                print(f"    Wayback: Found snapshot from {snapshot_date}")

                # Step 2: Fetch archived page content
                print(f"    Wayback: Fetching archived page content...")
                page_resp = await client.get(snapshot_url, headers=headers, timeout=25.0)

                if page_resp.status_code == 429:
                    wait = 2 ** attempt
                    print(f"    Wayback: HTTP 429 on page fetch. Sleeping {wait}s...")
                    await asyncio.sleep(wait)
                    continue

                if page_resp.status_code != 200:
                    status_msg = f"HTTP {page_resp.status_code} fetching snapshot content"
                    print(f"    Wayback: {status_msg}")
                    return {
                        "snapshot_date": snapshot_date,
                        "wayback_url": snapshot_url,
                        "wayback_status": status_msg,
                        "extracted_text": ""
                    }

                raw_html = page_resp.text

                # Step 3: Strip HTML to plain text
                text = re.sub(r'<!-- BEGIN WAYBACK TOOLBAR.*?END WAYBACK TOOLBAR -->', '', raw_html, flags=re.DOTALL)
                text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()

                extracted = text[:5000] if len(text) > 5000 else text
                print(f"    Wayback: Extracted {len(extracted)} chars of historical text")

                return {
                    "snapshot_date": snapshot_date,
                    "wayback_url": snapshot_url,
                    "wayback_status": "OK",
                    "extracted_text": extracted
                }

        except httpx.TimeoutException:
            wait = 2 ** attempt
            print(f"    Wayback: Timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait}s...")
            await asyncio.sleep(wait)
            continue
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 2 ** attempt
                print(f"    Wayback: HTTP 429 Rate Limited. Sleeping {wait}s...")
                await asyncio.sleep(wait)
                continue
            msg = f"HTTP {e.response.status_code} from CDX API"
            print(f"    Wayback: {msg}")
            return {
                "snapshot_date": "N/A",
                "wayback_url": "N/A",
                "wayback_status": msg,
                "extracted_text": ""
            }
        except Exception as e:
            msg = f"Error: {str(e)[:100]}"
            print(f"    Wayback Error for {url}: {e}")
            return {
                "snapshot_date": "N/A",
                "wayback_url": "N/A",
                "wayback_status": msg,
                "extracted_text": ""
            }

    # All retries exhausted
    return {
        "snapshot_date": "N/A",
        "wayback_url": "N/A",
        "wayback_status": f"Failed after {max_retries} retries (rate limited or timed out)",
        "extracted_text": ""
    }

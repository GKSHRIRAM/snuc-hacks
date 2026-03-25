import httpx
from typing import List
from urllib.parse import quote_plus

async def _search_via_ddg_html(query: str) -> List[str]:
    """
    Scrapes DuckDuckGo HTML results directly via httpx.
    No dependency on the broken `duckduckgo_search` package.
    """
    urls = []
    try:
        encoded = quote_plus(query)
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://html.duckduckgo.com/html/?q={encoded}",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10.0,
                follow_redirects=True
            )
            if resp.status_code == 200:
                text = resp.text
                # Extract result URLs from DDG HTML
                import re
                # DDG html results have links in <a class="result__a" href="...">
                matches = re.findall(r'class="result__a"[^>]*href="([^"]+)"', text)
                for m in matches[:3]:
                    # DDG wraps URLs in a redirect, extract the actual URL
                    if "uddg=" in m:
                        from urllib.parse import unquote, parse_qs, urlparse
                        parsed = urlparse(m)
                        qs = parse_qs(parsed.query)
                        if "uddg" in qs:
                            urls.append(unquote(qs["uddg"][0]))
                    elif m.startswith("http"):
                        urls.append(m)
    except Exception as e:
        print(f"  DDG HTML search error for '{query}': {e}")
    return urls

async def search_competitors_with_searxng(search_query: str) -> List[str]:
    """
    Agent Tool: Discovery agent.
    Splits complex OR queries into individual simple searches.
    Uses direct DDG HTML scraping (no broken pip packages).
    """
    all_urls: List[str] = []

    # Split complex OR queries into simpler individual ones
    parts = search_query.replace("(", "").replace(")", "").replace("site:.com", "").strip()
    sub_queries: List[str] = [q.strip() for q in parts.split(" OR ") if q.strip()]

    if len(sub_queries) <= 1:
        sub_queries = [search_query]

    for q in sub_queries:
        urls: List[str] = await _search_via_ddg_html(q)
        all_urls.extend(urls)

    # Deduplicate
    seen: set[str] = set()
    unique: List[str] = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    return [u for i, u in enumerate(unique) if i < 1]

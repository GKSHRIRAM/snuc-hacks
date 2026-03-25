from pygooglenews import GoogleNews
import httpx
import asyncio
from bs4 import BeautifulSoup

async def fetch_gdelt(domain):
    """Fetch recent mentions of the domain from GDELT using the DOC API."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": f'"{domain}"',
        "mode": "artlist",
        "format": "json",
    }
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url, params=params, timeout=20.0)
            res.raise_for_status()
            data = res.json()
            return data.get('articles', [])
    except Exception as e:
        print(f"GDELT error: {e}")
        return []

async def fetch_google_news(domain):
    """Fetch news articles using pygooglenews."""
    try:
        gn = GoogleNews()
        search = gn.search(domain)
        entries = search.get('entries', [])
        return [{'title': e.title, 'link': e.link, 'published': e.published} for e in entries[:20]]
    except Exception as e:
        print(f"Google News error: {e}")
        return []

async def collect_news(domain):
    """Aggregate news from Google News and GDELT."""
    gn_task = fetch_google_news(domain)
    gdelt_task = fetch_gdelt(domain)
    
    gn_results, gdelt_results = await asyncio.gather(gn_task, gdelt_task)
    
    return {
        "google_news": gn_results,
        "gdelt": gdelt_results
    }

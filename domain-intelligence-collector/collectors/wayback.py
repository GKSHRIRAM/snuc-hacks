import httpx
import asyncio

async def collect_wayback(domain):
    """Fetch historical snapshots from the Wayback Machine CDX API."""
    url = "http://web.archive.org/cdx/search/cdx"
    params = {
        "url": f"*.{domain}/*",
        "output": "json",
        "fl": "timestamp,original,statuscode,mimetype",
        "collapse": "urlkey",
        "limit": 100
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            if not data or len(data) <= 1:
                return []
                
            headers = data[0]
            results = []
            for row in data[1:]:
                results.append(dict(zip(headers, row)))
            return results
    except Exception as e:
        print(f"Wayback collection error: {e}")
        return []

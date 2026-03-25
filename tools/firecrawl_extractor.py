import os
import httpx

async def extract_markdown_with_firecrawl(url: str) -> str:
    """
    Agent Tool: Use this tool to hit the Firecrawl API and extract clean markdown from a live URL.
    Only use this if the Wayback Machine snapshot is missing or outdated.
    """
    api_key = os.environ.get("FIRECRAWL_API_KEY", "mock-firecrawl-key")
    firecrawl_url = "https://api.firecrawl.dev/v1/scrape"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "url": url,
        "formats": ["markdown"]
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(firecrawl_url, headers=headers, json=payload, timeout=30.0)
            
            if response.status_code == 403:
                # Agent instruction: Trigger FlareSolverr or stealth-browser fallback here
                return f"[BLOCKED] 403 Forbidden for {url}. Require FlareSolverr bypass."
                
            response.raise_for_status()
            data = response.json()
            
            return data.get("data", {}).get("markdown", "")
    except httpx.HTTPError as e:
        print(f"Firecrawl HTTP Error for {url}: {e}")
        return f"[ERROR] Network error scraping {url}: {str(e)}"
    except Exception as e:
        print(f"Firecrawl Error for {url}: {e}")
        return f"[ERROR] Failed to scrape {url}: {str(e)}"

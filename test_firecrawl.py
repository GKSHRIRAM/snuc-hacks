import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

async def test_firecrawl():
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key or api_key == "your_firecrawl_key_here":
        print("[-] FIrecrawl API key not set properly in .env")
        return

    url = "https://www.notion.com/"
    print(f"[*] Testing Firecrawl scraping against {url}...")
    
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
            print(f"[*] HTTP Status: {response.status_code}")
            print(f"[*] Response Body: {response.text[:1000]}...") # Print first 1000 chars
            
            if response.status_code == 200:
                data = response.json()
                md = data.get("data", {}).get("markdown", "")
                print(f"[+] Successfully extracted {len(md)} chars of Markdown!")
            
    except Exception as e:
        print(f"[-] Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_firecrawl())

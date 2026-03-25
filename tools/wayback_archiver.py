import httpx
from typing import Optional

async def archive_to_wayback(url: str) -> Optional[str]:
    """
    Agent Tool: Forces the Wayback Machine to take a live snapshot of the target URL 
    using the Internet Archive's "Save Page Now" (SPN) API.
    Returns the permanent archive link as verifiable proof.
    
    This turns MarketLens into a zero-storage intelligence engine:
    - Heavy HTML is stored for free on the Internet Archive
    - Our DB only stores the lightweight TOON summary + the archive proof URL
    """
    save_url = f"https://web.archive.org/save/{url}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(save_url, headers=headers, timeout=30.0)
            
            # The Wayback Machine returns the permanent link in Content-Location header
            if "Content-Location" in response.headers:
                archive_link = "https://web.archive.org" + response.headers["Content-Location"]
                print(f"    Archived: {archive_link}")
                return archive_link
            
            # Sometimes the redirect URL itself is the archive link
            if "web.archive.org/web/" in str(response.url):
                archive_link = str(response.url)
                print(f"    Archived (redirect): {archive_link}")
                return archive_link
            
            # Fallback: check response headers for any archive indicator
            if response.status_code == 200:
                # Try to extract from the Link header
                link_header = response.headers.get("Link", "")
                if "web.archive.org" in link_header:
                    import re
                    match = re.search(r'<(https://web\.archive\.org/web/[^>]+)>', link_header)
                    if match:
                        archive_link = match.group(1)
                        print(f"    Archived (link header): {archive_link}")
                        return archive_link

            print(f"    Archive request sent (HTTP {response.status_code}), snapshot may be processing")
            return f"https://web.archive.org/web/*/{url}"
            
    except Exception as e:
        print(f"    Wayback Archive Error for {url}: {e}")
        return None

import asyncio
from playwright.async_api import async_playwright
from google_play_scraper import search, reviews, Sort
import snscrape.modules.reddit as snreddit

async def fetch_reddit_reviews(domain):
    """Scrape Reddit for mentions and discussions using snscrape."""
    results = []
    try:
        scraper = snreddit.RedditSearchScraper(domain)
        for i, item in enumerate(scraper.get_items()):
            if i > 20: break
            results.append({
                "url": item.url,
                "author": item.author,
                "content": item.body if hasattr(item, 'body') else item.title,
                "created": str(item.created)
            })
    except Exception as e:
        print(f"Reddit scrape error: {e}")
    return results

async def fetch_google_play_reviews(domain):
    """Search for the domain's app and fetch reviews."""
    try:
        # Simple heuristic: domain name without tld as app query
        app_name = domain.split('.')[0]
        search_results = search(app_name, lang="en", country="us")
        if not search_results:
            return []
            
        app_id = search_results[0]['appId']
        rvs, _ = reviews(
            app_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=10
        )
        return [{"app_id": app_id, "score": r.get('score'), "text": r.get('content')} for r in rvs]
    except Exception as e:
        print(f"Google Play error: {e}")
        return []

async def fetch_trustpilot(domain):
    """Headless scrape of Trustpilot using Playwright."""
    results = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            url = f"https://www.trustpilot.com/review/{domain}"
            
            response = await page.goto(url, wait_until="domcontentloaded")
            if response and response.status == 200:
                # Extract basic text content (this can be refined)
                reviews_els = await page.locator("p[data-service-review-text-typography='true']").all_text_contents()
                results = [{"source": "trustpilot", "text": text} for text in reviews_els[:10]]
            await browser.close()
    except Exception as e:
        print(f"Trustpilot Playwright err: {e}")
        
    return results

async def collect_reviews(domain):
    reddit_task = fetch_reddit_reviews(domain)
    # Using run_in_executor for synchronous function
    play_reviews = await fetch_google_play_reviews(domain)
    tp_task = fetch_trustpilot(domain)
    
    reddit_res, tp_res = await asyncio.gather(reddit_task, tp_task)
    
    return {
        "reddit": reddit_res,
        "google_play": play_reviews,
        "trustpilot": tp_res
    }

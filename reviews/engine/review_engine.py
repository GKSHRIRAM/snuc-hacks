import requests
from bs4 import BeautifulSoup
import json
import time
import random

class ReviewEngine:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    def _clean_domain(self, domain):
        return (
            domain.lower()
            .replace("https://", "")
            .replace("http://", "")
            .replace("www.", "")
            .strip()
        )

    def _get_page_data(self, domain):
        """
        Step 1: Fetch the Trustpilot page and extract the __NEXT_DATA__ JSON blob.
        Returns businessUnitId, trustScore, and first-page reviews all at once.
        """
        url = f"https://www.trustpilot.com/review/{domain}"
        response = requests.get(url, headers=self.headers, timeout=10)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch page: HTTP {response.status_code}")

        soup = BeautifulSoup(response.text, "html.parser")
        script_tag = soup.find("script", {"id": "__NEXT_DATA__"})

        if not script_tag:
            raise Exception("__NEXT_DATA__ not found — Trustpilot may have changed structure")

        data = json.loads(script_tag.string)
        props = data["props"]["pageProps"]

        business_unit = props["businessUnit"]
        business_unit_id = business_unit["id"]
        trust_score = float(business_unit["trustScore"])

        # First page of reviews is already embedded in the HTML
        first_page_reviews = props.get("reviews", [])

        return business_unit_id, trust_score, first_page_reviews

    def _fetch_reviews_page(self, business_unit_id, page):
        """
        Step 2: Hit Trustpilot's internal API for subsequent pages.
        This is the same endpoint their frontend uses — returns clean JSON.
        """
        url = f"https://www.trustpilot.com/api/categoriespages/businessunits/{business_unit_id}/reviews"
        params = {
            "page": page,
            "perPage": 20,
            "language": "en",
        }
        response = requests.get(url, headers=self.headers, params=params, timeout=10)

        if response.status_code == 200:
            return response.json().get("reviews", [])
        return []

    def _parse_reviews(self, raw_reviews):
        """
        Normalize raw review objects into clean dicts your app can use.
        Handles both __NEXT_DATA__ format and API format (slightly different keys).
        """
        parsed = []
        for r in raw_reviews:
            # __NEXT_DATA__ uses "text", API uses "text" too — but nesting differs
            body = r.get("text") or r.get("body", "")
            rating = r.get("rating") or r.get("stars", 0)
            title = r.get("title", "")

            if body and len(body) > 20:
                parsed.append({
                    "body": body,
                    "title": title,
                    "rating": rating,
                })
        return parsed

    def get_review_data(self, domain, max_pages=3):
        """
        Main entry point. Returns (sentiment_score, reviews list).
        max_pages controls how many pages beyond page 1 to fetch.
        """
        clean = self._clean_domain(domain)
        all_reviews = []
        final_score = 0.0

        try:
            # Step 1 — one request gets us everything for page 1
            business_unit_id, trust_score, first_page = self._get_page_data(clean)

            # Normalize: (stars - 2.5) / 2.5 centers neutral at 2.5
            final_score = (trust_score - 2.5) / 2.5

            all_reviews.extend(self._parse_reviews(first_page))

            # Step 2 — fetch additional pages from internal API
            for page in range(2, max_pages + 1):
                time.sleep(random.uniform(0.8, 1.8))  # be polite
                page_reviews = self._fetch_reviews_page(business_unit_id, page)
                if not page_reviews:
                    break  # no more pages
                all_reviews.extend(self._parse_reviews(page_reviews))

        except Exception as e:
            print(f"[ReviewEngine] Error: {e}")
            return 0.0, []
        
        return final_score, all_reviews

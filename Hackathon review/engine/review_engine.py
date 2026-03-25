import requests
import os

class ReviewEngine:
    def __init__(self):
        self.api_key = os.getenv("RAPID_API_KEY")

    def get_g2_score(self, company):
        # Using the G2 Data API from RapidAPI (Free Tier)
        url = "https://g2-data-api.p.rapidapi.com/search" 
        headers = {"X-RapidAPI-Key": self.api_key}
        
        try:
            response = requests.get(url, headers=headers, params={"q": company}, timeout=10)
            data = response.json()
            # Normalize a 5-star rating to a -1 to 1 sentiment scale
            rating = data.get('reviews', [{}])[0].get('rating', 3) 
            return (rating - 3) / 2 
        except Exception:
            return 0
import requests
import os

class ReviewEngine:
    def __init__(self):
        self.api_key = os.getenv("RAPID_API_KEY")
        self.host = "g2-data-api.p.rapidapi.com"

    def get_review_score(self, product_name):
        url = f"https://{self.host}/g2-products"
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host,
            "Content-Type": "application/json"
        }
        params = {"product": product_name, "max_reviews": "100"}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            data = response.json()
            
            # DIGGING INTO THE JSON: G2 returns a list under 'data'
            results = data.get('data', [])
            if results:
                # Use the first match's rating (usually 1.0 - 5.0)
                rating = results[0].get('rating', 3.0)
                # Normalize: (Rating - 3) / 2 converts 1-5 scale to -1 to 1
                return (float(rating) - 3) / 2
            return 0.0
        except Exception as e:
            print(f"G2 Error: {e}")
            return 0.0

import praw
import os
from transformers import pipeline

class RedditEngine:
    def __init__(self):
        # This downloads a small AI model (~260MB) on first run
        self.analyzer = pipeline("sentiment-analysis", 
                                 model="distilbert-base-uncased-finetuned-sst-2-english")
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="MarketIntel/1.0"
        )

    def get_sentiment(self, query):
        try:
            posts = [s.title for s in self.reddit.subreddit("all").search(query, limit=15)]
            if not posts: return 0.0
            
            results = self.analyzer(posts)
            # Positive = 1, Negative = -1
            scores = [1 if r['label'] == 'POSITIVE' else -1 for r in results]
            return sum(scores) / len(scores)
        except Exception:
            return 0.0

import praw
import os
from transformers import pipeline

class RedditEngine:
    def __init__(self):
        # This downloads a small model to run locally on your friend's PC
        self.analyzer = pipeline("sentiment-analysis", 
                                 model="distilbert-base-uncased-finetuned-sst-2-english")
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="BTechStudentProject/1.0"
        )

    def get_sentiment(self, query):
        # Search all of Reddit for the keyword
        titles = [s.title for s in self.reddit.subreddit("all").search(query, limit=25)]
        if not titles: return 0
        
        results = self.analyzer(titles)
        # Convert POSITIVE/NEGATIVE labels to -1 to 1 scale
        scores = [1 if r['label'] == 'POSITIVE' else -1 for r in results]
        return sum(scores) / len(scores)
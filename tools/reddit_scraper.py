import os
import praw

def get_reddit_sentiment_sync(competitor_name: str) -> str:
    """
    Synchronous PRAW integration to fetch top 5 recent posts/complaints about a competitor.
    """
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = "MarketLensBI/1.0"

    if not client_id or not client_secret:
        return "Insufficient Data (REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET missing)"

    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

        query = f"{competitor_name} (issue OR bug OR complaint OR sucks OR review OR missing)"
        
        posts_text = []
        # Search all of reddit
        for submission in reddit.subreddit("all").search(query, sort="relevance", limit=5):
            title = submission.title.strip()
            selftext = submission.selftext.strip()[:300] # trunc
            
            post_str = f"- Title: {title}"
            if selftext:
                post_str += f"\n  Body: {selftext}..."
            posts_text.append(post_str)

        if not posts_text:
            return "Insufficient Data (No relevant Reddit complaints found)"

        return "\n".join(posts_text)

    except Exception as e:
        return f"Insufficient Data (Reddit API Error: {str(e)[:100]})"

import tweepy
import pandas as pd
import os

# ==============================
# 1. Twitter API Authentication
# ==============================
# Replace with your keys from https://developer.x.com/en/portal/dashboard
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAACv53gEAAAAAiJBkIk89hWXmaE0yaS0mKgWoHcc%3DMmwXEgeGLGhr4Qsb4VjF0h4mpEgZq840WEqQ0NQRxmlOzixiqP"


client = tweepy.Client(bearer_token=BEARER_TOKEN)

# ==============================
# 2. Function to collect tweets
# ==============================
def collect_tweets(query, max_results=50):
    tweets = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=["id", "text", "created_at", "lang"]
    )
    
    tweet_data = []
    if tweets.data:
        for tweet in tweets.data:
            tweet_data.append([tweet.id, tweet.text, tweet.created_at, tweet.lang])
    
    # Save to CSV
    df = pd.DataFrame(tweet_data, columns=["id", "text", "created_at", "lang"])
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/live_tweets.csv", index=False, encoding="utf-8")
    print(f"✅ Collected {len(df)} tweets for '{query}' and saved to data/raw/live_tweets.csv")


if __name__ == "__main__":
    keyword = input("Enter keyword to search: ")
    collect_tweets(keyword, max_results=50)

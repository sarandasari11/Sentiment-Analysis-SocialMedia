import tweepy
import pandas as pd
import yaml

# load keys
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

bearer_token = config["bearer_token"]
client = tweepy.Client(bearer_token=bearer_token)

def fetch_tweets(query, max_results=50):
    tweets = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=["created_at","lang"])
    data = [{"text": tweet.text, "created_at": tweet.created_at} for tweet in tweets.data]
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = fetch_tweets("Tesla lang:en -is:retweet", max_results=50)
    df.to_csv("data/raw/tesla_tweets.csv", index=False)
    print("Tweets saved to data/raw/tesla_tweets.csv")

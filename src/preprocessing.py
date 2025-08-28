import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# download stopwords first time
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Cleans a single tweet/text string"""
    text = text.lower()
    text = re.sub(r"@\w+", "", text)          # remove mentions
    text = re.sub(r"#\w+", "", text)          # remove hashtags
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"[^\w\s]", "", text)       # remove punctuation
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

def preprocess_sentiment140(input_file, output_file):
    """Cleans Sentiment140 dataset"""
    df = pd.read_csv(input_file, encoding="latin-1", header=None)
    df.columns = ["target", "ids", "date", "flag", "user", "text"]

    # keep only target + text
    df = df[["target", "text"]]

    # map targets
    df["target"] = df["target"].map({0:"negative", 2:"neutral", 4:"positive"})

    # clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # save processed dataset
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned Sentiment140 saved to {output_file}")

def preprocess_live_tweets(input_file, output_file):
    """Cleans tweets collected from API"""
    df = pd.read_csv(input_file)
    df["clean_text"] = df["text"].apply(clean_text)

    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned live tweets saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    preprocess_sentiment140("data/raw/sentiment140.csv", "data/processed/sentiment140_clean.csv")
    preprocess_live_tweets("data/raw/tesla_tweets.csv", "data/processed/tesla_tweets_clean.csv")

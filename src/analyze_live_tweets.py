import pandas as pd
import re
import joblib   
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


# ==============================
# 1. Load trained model + vectorizer
# ==============================
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# ==============================
# 2. Preprocess function
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r"@\w+|#\w+", "", text)  # remove mentions/hashtags
    text = re.sub(r"[^a-z\s]", "", text)  # remove special chars/numbers
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# ==============================
# 3. Load live tweets
# ==============================
df = pd.read_csv("data/raw/live_tweets.csv")

# Preprocess
df["clean_text"] = df["text"].astype(str).apply(clean_text)

# ==============================
# 4. Transform + Predict
# ==============================
X = vectorizer.transform(df["clean_text"])
df["predicted_sentiment"] = model.predict(X)

# Map labels
df["sentiment_label"] = df["predicted_sentiment"].map({0: "negative", 1: "positive"})

# ==============================
# 5. Save results
# ==============================
df.to_csv("data/processed/live_tweets_classified.csv", index=False, encoding="utf-8")

# Show quick trend summary
summary = df["sentiment_label"].value_counts(normalize=True) * 100
print("✅ Sentiment distribution (%):")
print(summary)
print("✅ Classified tweets saved to data/processed/live_tweets_classified.csv")
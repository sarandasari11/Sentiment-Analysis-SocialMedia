import joblib

# Load saved model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Test sentences
samples = [
    "I love Tesla cars, they are amazing!",
    "The new update is terrible and I hate it",
    "This is okay, nothing special",
    "I'm so happy with the new features",
    "Worst experience ever"
]

# Transform and predict
X = vectorizer.transform(samples)
preds = model.predict(X)

for text, label in zip(samples, preds):
    sentiment = "Positive 😀" if label == 1 else "Negative 😡"
    print(f"{text} -> {sentiment}")

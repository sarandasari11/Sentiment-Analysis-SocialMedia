import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib   # to save model

def train_model(input_file, model_file, vectorizer_file):
    # Load preprocessed Sentiment140
    df = pd.read_csv(input_file)

    # Drop rows with missing values
    df = df.dropna(subset=["clean_text"])

    # Ensure text is string type
    X = df["clean_text"].astype(str)
    y = df["target"]

    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    # Train Logistic Regression
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save model + vectorizer
    joblib.dump(clf, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    print(f"✅ Model saved to {model_file}")
    print(f"✅ Vectorizer saved to {vectorizer_file}")



if __name__ == "__main__":
    train_model(
        "data/processed/sentiment140_clean.csv",
        "models/sentiment_model.pkl",
        "models/tfidf_vectorizer.pkl"
    )

import pandas as pd
import joblib

def predict_sentiment(input_file, model_file, vectorizer_file, output_file):
    # Load model + vectorizer
    clf = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)

    # Load preprocessed live tweets
    df = pd.read_csv(input_file)

    # Transform text
    X_vec = vectorizer.transform(df["clean_text"])

    # Predict
    df["predicted_sentiment"] = clf.predict(X_vec)

    # Save results
    df.to_csv(output_file, index=False)
    print(f"✅ Predictions saved to {output_file}")

if __name__ == "__main__":
    predict_sentiment(
        "data/processed/tesla_tweets_clean.csv",
        "models/sentiment_model.pkl",
        "models/tfidf_vectorizer.pkl",
        "outputs/tesla_predictions.csv"
    )

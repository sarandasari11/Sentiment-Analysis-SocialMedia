import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# Load preprocessed data
df = pd.read_csv("data/processed/sentiment140_clean.csv")
print(df.head())  # 👀 Check columns

# Drop NaN or empty texts
df = df.dropna(subset=["clean_text"])
df = df[df["clean_text"].str.strip() != ""]

# Features and labels
X = df["clean_text"]
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------
# Logistic Regression Model
# -------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_vec, y_train)

y_pred_log = log_model.predict(X_test_vec)

print("\n🔹 Logistic Regression Classification Report:\n")
print(classification_report(y_test, y_pred_log))

cm_log = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save Logistic Regression model
joblib.dump(log_model, "models/logistic_regression_model.pkl")

# -------------------------
# Naive Bayes Model
# -------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

y_pred_nb = nb_model.predict(X_test_vec)

print("\n🔹 Naive Bayes Classification Report:\n")
print(classification_report(y_test, y_pred_nb))

cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save Naive Bayes model
joblib.dump(nb_model, "models/naive_bayes_model.pkl")

# Save vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("✅ Both models & Vectorizer saved with joblib")

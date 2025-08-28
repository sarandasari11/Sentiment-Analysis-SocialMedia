import streamlit as st
import pandas as pd
import joblib

# --- Page Config ---
st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    layout="wide",
    page_icon="📊"
)

# --- Title and Info ---
st.title(" Social Media Sentiment Dashboard")
st.markdown("""
Welcome! This dashboard allows you to analyze the sentiment of social media posts such as tweets, Facebook comments, or Instagram posts.  

You can either:
1. **Test a single sentence** in the box below.  
2. **Upload a CSV file** containing a column named `text,date`.  

The models classify text as **Positive** or **Negative** based on pre-trained machine learning models.
""")

# --- Load Models ---
log_reg = joblib.load("../models/logistic_regression_model.pkl")
nb_model = joblib.load("../models/naive_bayes_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

# --- Sidebar ---
st.sidebar.title("Settings")
model_choice = st.sidebar.radio("Select Model", ("Logistic Regression", "Naive Bayes"))
uploaded_file = st.sidebar.file_uploader("Upload CSV (column 'text')", type=["csv"])
single_text = st.sidebar.text_area("Test Single Text:")

# --- Select model ---
model = log_reg if model_choice == "Logistic Regression" else nb_model

# --- Helper function ---
def label_to_sentiment(label):
    return "Positive" if label == 1 else "Negative"

# --- Single Text Prediction ---
st.subheader(" Single Text Prediction")
st.info("Type a sentence (e.g., 'Tesla is amazing') and click the button to see the predicted sentiment.")

# Initialize session state
if "single_text_pred" not in st.session_state:
    st.session_state.single_text_pred = ""

vec = None
pred = None

if st.button("Predict Sentiment"): 
    if single_text.strip() != "": 
        vec = vectorizer.transform([single_text]) 
        pred = model.predict(vec)[0] 
        st.session_state.single_text_pred = pred 
        # Display prediction 
        if st.session_state.single_text_pred != "":
            st.success(f"Predicted Sentiment:{st.session_state.single_text_pred}")
    
# --- CSV Predictions ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("CSV must have a column named 'text'")
    else:
        # Predict sentiment for CSV
        df["predicted_label"] = model.predict(vectorizer.transform(df["text"].astype(str)))
        

        st.subheader("Predictions (First 500 rows)")
        st.dataframe(df.head(500))

        # --- Bar Chart ---
        st.subheader("Sentiment Counts")
        st.bar_chart(df["predicted_label"].value_counts())

        # --- Download Predictions ---
        st.subheader("⬇ Download Predictions")
        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_out,
            file_name="predictions.csv",
            mime="text/csv"
        )

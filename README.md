# 💬 Social Media Sentiment Analysis Dashboard

## 📌 Project Overview  
Social media platforms generate massive amounts of user opinions every second.  
Understanding sentiment in these posts helps in:  
- 📢 **Brand Monitoring** → Track public opinion about products/services  
- 📰 **Trend Analysis** → Identify viral topics & customer concerns  
- 🎯 **Decision Support** → Assist businesses in making data-driven marketing strategies  

This project applies **Machine Learning & NLP techniques** to classify posts into **Positive / Negative / Neutral** sentiments and provides an **interactive Streamlit dashboard** for real-time analysis.

---

## ⚙️ Features
✅ End-to-End Pipeline:  
- Text preprocessing (stopword removal, stemming, cleaning hashtags & mentions)  
- Feature extraction with **TF-IDF**  
- ML Models: **Logistic Regression** & **Naive Bayes**  
- Performance evaluation with **Accuracy, Precision, Recall, F1-score**  

✅ Streamlit Dashboard with:  
- Single text sentiment prediction  
- Bulk CSV sentiment analysis  
- Sentiment distribution visualization (charts)  
- Downloadable results as CSV  

✅ Jupyter Notebook for model training & evaluation  

✅ Clean results & comparison plots  

---

## 📂 Project Structure
```
Sentiment-Analysis-SocialMedia/
│── data/ # Dataset (sentiment140_clean.csv)
│── notebooks/ # Jupyter notebooks
│ └── sentiment_analysis.ipynb
│── models/ # Saved models
│ ├── logistic_regression_model.pkl
│ ├── naive_bayes_model.pkl
│ └── tfidf_vectorizer.pkl
│── outputs/ # Model metrics & visualizations
│── dashboard/ # Streamlit dashboard app
│ └── app.py
│── requirements.txt # Dependencies
│── README.md # Project documentation
```

---

## 📝 File Descriptions  

- **data/sentiment140_clean.csv** → Preprocessed dataset for training & testing  

- **notebooks/sentiment_analysis.ipynb** →  
  • EDA & visualization of sentiment distribution  
  • Text preprocessing & feature extraction  
  • Training Logistic Regression & Naive Bayes models  
  • Model evaluation & saving trained models  

- **models/** → Stores trained ML models & vectorizer  

- **outputs/** → Contains metrics reports & confusion matrix plots  

- **dashboard/app.py** →  
  • Streamlit app for interactive sentiment prediction  
  • Supports both single text & CSV upload  
  • Displays sentiment distribution charts  
  • Option to download predictions  

- **requirements.txt** → Python dependencies  

---

## 📊 Model Comparison  
- **Logistic Regression** → Higher accuracy for short texts  
- **Naive Bayes** → Efficient for large-scale text classification  
- Both models compared using **Accuracy, Precision, Recall, F1-score**  

---

## 🚀 How to Run
##  Dataset Download

The project uses a preprocessed version of the Sentiment140 dataset, which includes simulated dates and is ready for modeling.

** Download Link:**  
[Download the cleaned sentiment dataset (CSV)](https://drive.google.com/file/d/1FgKKo0vMkg3s1qCHSCYOjwe7VWdkf4UD/view?usp=sharing)

** Instructions:**
1. Click the link above to open the Google Drive file.
2. Click the **Download** (⤓) button to save the file to your computer.
3. Move the downloaded file into the project’s `data/processed/` directory.
4. Rename it to:  

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/sarandasari11/Sentiment-Analysis-SocialMedia.git
cd Sentiment-Analysis-SocialMedia

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

data/sentiment140_clean.csv

cd dashboard
streamlit run app.py

```
---
📈 Results & Insights

- Logistic Regression performed better on short informal tweets

- Naive Bayes was faster and worked well on large datasets

- Dashboard allows real-time testing of posts and batch analysis via CSV

🔮 Future Improvements

- Add Deep Learning (LSTM, BERT) models

- Extend to multilingual sentiment analysis

- Deploy with Docker + Cloud (AWS/GCP)

- Add topic modeling (LDA) for deeper insights

## 👨‍💻 Author
**Saran Dasari**  
📌 GitHub: [sarandasari11](https://github.com/sarandasari11)  
📧 Contact: dasarisaran2005@gmail.com

---
✨ Thank you for checking out this project!  
If you find it useful, please ⭐ the repo on GitHub.

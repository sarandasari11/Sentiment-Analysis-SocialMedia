import pandas as pd
import numpy as np

# Load preprocessed CSV
df = pd.read_csv("data/processed/sentiment140_clean.csv")

# Simulate random dates in the last 30 days
np.random.seed(42)
date_range = pd.date_range(end=pd.Timestamp.today(), periods=30).to_pydatetime().tolist()
df['date'] = np.random.choice(date_range, size=len(df))

# Save new CSV
df.to_csv("data/processed/sentiment140_with_dates.csv", index=False)
print("✅ CSV updated with simulated dates")

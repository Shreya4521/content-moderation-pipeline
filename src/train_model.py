"""
train_model.py
Trains a TF-IDF + Logistic Regression toxicity classifier and saves it as a pickle.

Run this once before starting the app:
    python src/train_model.py

To use a bigger real-world dataset instead of the bundled sample:
1. Download the Jigsaw Toxic Comment Classification dataset from Kaggle:
   https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
2. Place train.csv in the data/ folder
3. Update DATA_PATH and TEXT_COL/LABEL_COL below
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Toggle this to switch between the bundled sample dataset and the Kaggle Jigsaw dataset ---
USE_KAGGLE_DATASET = False

SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_dataset.csv")
KAGGLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Jigsaw dataset has 6 separate label columns instead of one; we merge them into a single
# binary "toxic vs safe" label since our rules.py + app.py pipeline is built for binary output.
JIGSAW_LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def load_data():
    if USE_KAGGLE_DATASET:
        df = pd.read_csv(KAGGLE_DATA_PATH)
        df = df.rename(columns={"comment_text": "text"})
        # Row is "toxic" (1) if ANY of the 6 label columns are 1, else "safe" (0)
        df["label"] = df[JIGSAW_LABEL_COLS].max(axis=1)
        df = df[["text", "label"]]
    else:
        df = pd.read_csv(SAMPLE_DATA_PATH)
    return df


TEXT_COL = "text"
LABEL_COL = "label"


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COL], df[LABEL_COL], test_size=0.25, random_state=42, stratify=df[LABEL_COL]
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=["safe", "toxic"]))

    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, "toxicity_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel and vectorizer saved to {MODEL_DIR}/")


if __name__ == "__main__":
    train()

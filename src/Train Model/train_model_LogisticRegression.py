# src/train_logistic.py

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

train_df = pd.read_csv("data/processed/english/train.csv")
test_df = pd.read_csv("data/processed/english/test.csv")

X_train = train_df["clean_text"]
y_train = train_df["target"]

X_test = test_df["clean_text"]
y_test = test_df["target"]

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,2)
        )
    ),
    (
        "logistic",
        LogisticRegression(
            max_iter=1000
        )
    )
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nF1 Score:")
print(f1_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

joblib.dump(
    model,
    "models/logistic_regression/logistic_model.pkl"
)
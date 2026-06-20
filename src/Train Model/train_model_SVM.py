# src/train_svm.py

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

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
        "svm",
        SVC(
            kernel="linear"
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

joblib.dump(model, "models/SVM/svm_model.pkl")
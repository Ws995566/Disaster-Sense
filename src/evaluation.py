import pandas as pd
import numpy as np
import torch
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# LOAD TEST DATA

test_df = pd.read_csv(
    "data/processed/English/test.csv"
)

X_test = test_df["clean_text"].tolist()
y_test = test_df["target"].tolist()

results = []

# HELPER FUNCTION

def evaluate_model(
    model_name,
    predictions,
    labels
):

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions),
        "Recall": recall_score(labels, predictions),
        "F1 Score": f1_score(labels, predictions)
    })


# LOGISTIC REGRESSION

logistic_model = joblib.load(
    "models/logistic_regression/logistic_model.pkl"
)

logistic_pred = logistic_model.predict(X_test)

evaluate_model(
    "Logistic Regression",
    logistic_pred,
    y_test
)

print("Logistic Regression Done")


# SVM

svm_model = joblib.load(
    "models/svm/svm_model.pkl"
)

svm_pred = svm_model.predict(X_test)

evaluate_model(
    "SVM",
    svm_pred,
    y_test
)

print("SVM Done")


# TRANSFORMER EVALUATION FUNCTION

def evaluate_transformer(
    model_path,
    tokenizer_path,
    model_name
):

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    predictions = []

    for text in X_test:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():

            outputs = model(**inputs)

            pred = torch.argmax(
                outputs.logits,
                dim=1
            ).item()

        predictions.append(pred)

    evaluate_model(
        model_name,
        predictions,
        y_test
    )

    print(f"{model_name} Done")


# DISTILBERT

evaluate_transformer(
    "models/distilbert",
    "distilbert-base-uncased",
    "DistilBERT"
)

# BERT

evaluate_transformer(
    "models/english_bert",
    "bert-base-uncased",
    "BERT"
)

# ROBERTA

evaluate_transformer(
    "models/roberta",
    "roberta-base",
    "RoBERTa"
)

# ALBERT

evaluate_transformer(
    "models/albert-base-v2",
    "albert-base-v2",
    "ALBERT"
)

# SAVE RESULTS

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print(results_df)


from transformers import pipeline
from src.language_detection import detect_language

english_classifier = pipeline("text-classification", model="MemoriesW/DisasterSense-English")

indo_classifier = pipeline("text-classification", model="MemoriesW/DisasterSense-Indonesian")

label_mapping = {"LABEL_0": "Non-Disaster", "LABEL_1": "Disaster"}

def predict_disaster(text):

    language = detect_language(text)

    if language == "id":
        result = indo_classifier(text)

    else:
        result = english_classifier(text)

    prediction = result[0]["label"]
    confidence = result[0]["score"]
    prediction = label_mapping.get(prediction, prediction)

    return {
        "language": language,
        "prediction": prediction,
        "confidence": confidence
    }
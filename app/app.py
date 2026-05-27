import sys
import os

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import streamlit as st

from src.inference import predict_disaster


st.title("Disaster Sense")

user_input = st.text_area(
    "Enter Tweet:"
)

if st.button("Analyze"):

    if user_input.strip() == "":

        st.warning("Please enter text.")

    else:

        result = predict_disaster(user_input)

        st.write("Language:",result["language"])
        st.write("Prediction:",result["prediction"])
        st.write("Confidence:",round(result["confidence"] * 100,2),"%")
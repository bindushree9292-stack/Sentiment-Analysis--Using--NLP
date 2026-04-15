import streamlit as st
import pickle
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import clean_text

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("💬 Sentiment Analysis")
st.write("Analyze text sentiment using Machine Learning")

text = st.text_area("Enter your text here")

if st.button("Analyze Sentiment"):
    if text.strip() != "":
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        result = model.predict(vec)[0]

        if result == "positive":
            st.success("😊 Positive Sentiment")
        elif result == "negative":
            st.error("😠 Negative Sentiment")
        else:
            st.warning("😐 Neutral Sentiment")
    else:
        st.warning("Please enter text!")
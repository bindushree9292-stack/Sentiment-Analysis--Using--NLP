# Advanced Sentiment Analysis Project

## Features
- TF-IDF + Logistic Regression (trained on 75 balanced samples: 25 positive, 25 negative, 25 neutral)
- NLP preprocessing (NLTK)
- Streamlit Web App
- Twitter API integration

## Setup

pip install -r requirements.txt

## Train Model (uses enhanced data)

cd src
python train.py

## Run App

streamlit run app/app.py
# or python -m streamlit run app/app.py

## Test Model

python test_model.py
python -m streamlit run app/app.py

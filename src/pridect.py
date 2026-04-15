import pickle
from .preprocess import clean_text

model = pickle.load(open('../model.pkl', 'rb'))
vectorizer = pickle.load(open('../vectorizer.pkl', 'rb'))

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

if __name__ == "__main__":
    while True:
        text = input("Enter text: ")
        print("Sentiment:", predict_sentiment(text))
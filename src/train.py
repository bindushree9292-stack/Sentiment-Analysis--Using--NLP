import pandas as pd
import pickle
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from preprocess import clean_text

# Load dataset
data_path = os.path.join(script_dir, '..', 'data', 'enhanced_sample_data.csv')
df = pd.read_csv(data_path)

# Preprocess
df['cleaned'] = df['text'].apply(clean_text)

X = df['cleaned']
y = df['label']

# Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

# Model
model = LogisticRegression(max_iter=200)
model.fit(X_vec, y)

# Save
model_path = os.path.join(script_dir, '..', 'model.pkl')
vectorizer_path = os.path.join(script_dir, '..', 'vectorizer.pkl')
pickle.dump(model, open(model_path, 'wb'))
pickle.dump(vectorizer, open(vectorizer_path, 'wb'))

print("✅ Model trained successfully!")
import pandas as pd
import numpy as np
from keras.models import load_model
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model
model = load_model(r'C:\Users\C A I S T\Downloads\project\BPD_Sentinal.h5')

# Load the saved label encoder and TF-IDF vectorizer
label_encoder = joblib.load(r'C:\Users\C A I S T\Downloads\project\label_encoder.pkl')
vectorizer = joblib.load(r'C:\Users\C A I S T\Downloads\project\tfidf_vectorizer.pkl')

# Function to make predictions
def predict_subreddit(text):
    text_tfidf = vectorizer.transform([text]).toarray()
    prediction = model.predict(text_tfidf)
    predicted_class = np.argmax(prediction, axis=1)
    subreddit = label_encoder.inverse_transform(predicted_class)
    return subreddit[0]

# Example usage
test_text = "i hate this world"
predicted_subreddit = predict_subreddit(test_text)
print(f'The predicted subreddit is: {predicted_subreddit}')

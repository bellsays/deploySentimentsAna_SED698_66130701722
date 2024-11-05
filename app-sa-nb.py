

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import streamlit as st
import pickle

# Load the model using pickle
with open('sentiment_pipeline_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Set up the Streamlit app
st.title("Sentiment Analysis")

# Create an input box for user text
user_input = st.text_input("Enter a sentence for sentiment analysis:")

if st.button("Predict Sentiment"):
    if user_input:
        # Make a prediction
        predictions = loaded_model.predict([user_input])

        #sentiment = "Positive" if predictions[0] == 1 else "Negative"

        # Display the result
        st.write("Sentiment Prediction:", predictions)
    else:
        st.write("Please enter a sentence for analyze.")

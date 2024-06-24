import streamlit as st
import joblib
import pickle
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import nltk

# Ensure NLTK's punkt tokenizer is downloaded
nltk.download('punkt')

# Load the Word2Vec model
try:
    word2vec_model = Word2Vec.load("word2vec_modeltrialnb.model")
except FileNotFoundError:
    st.error("The Word2Vec model file 'word2vec_modeltrialnb.model' was not found.")
    st.stop()

# Load the Naive Bayes classifier
try:
    with open("classifier.pkl", "rb") as f:
        nb = pickle.load(f)
except FileNotFoundError:
    st.error("The classifier file 'classifier.pkl' was not found.")
    st.stop()

# Define sentiment labels
sentiment_labels = {'1': 'Medication', '0': 'Non-medication', '2': 'Others'}

# Function to create document vectors by averaging word vectors
def document_vector(model, doc):
    doc = [word for word in word_tokenize(str(doc)) if word in model.wv]
    if len(doc) == 0:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[doc], axis=0)

# Create Streamlit app
st.title("Medication Classification")

# Input text area
user_input = st.text_area('Enter your text here')

# Prediction button
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Convert input text to feature vector
        input_vector = document_vector(word2vec_model, user_input).reshape(1, -1)
        
        # Predict sentiment
        predicted_sentiment = nb.predict(input_vector)[0]
        predicted_sentiment_label = sentiment_labels[str(predicted_sentiment)]

        # Display predicted sentiment
        st.info(f"Predicted Sentiment: {predicted_sentiment_label}")

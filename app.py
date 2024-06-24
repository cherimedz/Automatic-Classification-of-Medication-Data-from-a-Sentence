import streamlit as st
import joblib
import pickle
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import nltk


nltk.download('punkt')


try:
    word2vec_model = Word2Vec.load("word2vec_modeltrialnb.model")
except FileNotFoundError:
    st.error("The Word2Vec model file 'word2vec_modeltrialnb.model' was not found.")
    st.stop()


try:
    with open("classifier.pkl", "rb") as f:
        nb = pickle.load(f)
except FileNotFoundError:
    st.error("The classifier file 'classifier.pkl' was not found.")
    st.stop()

labels = {'1': 'Medication', '0': 'Non-medication', '2': 'Others'}


def document_vector(model, doc):
    doc = [word for word in word_tokenize(str(doc)) if word in model.wv]
    if len(doc) == 0:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[doc], axis=0)


st.title("Medication Classification")


user_input = st.text_area('Enter your text here')


if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        
        input_vector = document_vector(word2vec_model, user_input).reshape(1, -1)
        
        
        predicted_class = nb.predict(input_vector)[0]
        predicted_class_label = labels[str(predicted_class)]

        
        st.info(f"Predicted Class of Comment: {predicted_class_label}")

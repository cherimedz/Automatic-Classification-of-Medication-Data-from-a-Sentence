import streamlit as st
import joblib
import pickle
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About Us", "Contact Us"])

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
if page == "Home":


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
elif page == "About Us":
    st.title("About Us")
    st.write("This is a NLP-based project that uses Word2Vec and Naive Bayes model.")
elif page == "Contact Us":
    st.title("Contact Us")
    st.write("Hi! We are the team behind this project. Wanna know more? Please feel free to contact:")
    st.write("Annapurna Padmanabhan")
    st.markdown("[Github ](https://github.com/annapurna1702)")
    st.markdown("[LinkedIn ](https://www.linkedin.com/in/annapurnapadmanabhan/)")
    st.write("Sourikta Nag")
    st.markdown("[Github ](https://github.com/srikta)")
    st.markdown("[LinkedIn ](https://www.linkedin.com/in/sourikta-nag-200106235?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)")
    st.write("Mansi")
    st.markdown("[Github ](https://github.com/MansiMalani)")
    st.markdown("[LinkedIn ](https://www.linkedin.com/in/mansi-malani-b4b0b327b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)")
    st.write("Medha Reju Pillai")
    st.markdown("[Github ](https://github.com/cherimedz)")
    st.markdown("[LinkedIn ](https://www.linkedin.com/in/medha-reju-pillai-42551b277?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)")
    st.write("Anju B J")
    st.markdown("[Github ](https://github.com/Anju-B-J)")
    st.markdown("[LinkedIn ](http://www.linkedin.com/in/anjubj)")


import streamlit as st
import pickle
import re

# load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# Streamlit UI
st.title("Sentiment Analysis App")

st.write("Enter a review and the model will predict sentiment.")

user_input = st.text_area("Enter Review")

if st.button("Predict Sentiment"):

    cleaned_text = clean_text(user_input)

    vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(vector)[0]

    st.success(f"Predicted Sentiment: {prediction}")
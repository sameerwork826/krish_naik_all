# making stremalit app for project deployment and working

# laoding the model

import numpy as np
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import load_model
import streamlit as st

# loading the model
model.load('simple_rnn_imdb.h5')


# load the word index
word_index=imdb.get_word_index()
reverse_word_index = {value:key for key, value in word_index.items()}


def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3."?") for i in encoded_review])

# preprocessing the input

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# prediction function

def prediction_sentiment(text):
    padded_review = preprocess_text(text)
    prediction = model.predict(padded_review)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment,prediction[0][0]


# streamlit app

st.title("Sentiment Analysis on IMDB Reviews")
st.write("Enter a movie review to predict its sentiment (Positive/Negative):")


user_input=st.text_area("Movie Review")

if st.button("classify"):
    
    preproces_input = preprocess_text(user_input)
    
    # make prediction
    prediction=model.predict(preproces_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    st.write(f"Sentiment: {sentiment} , probability: {prediction[0][0]:.2f}")
else:
    st.write("Please enter a review and click on 'classify' to see the sentiment prediction.")  
    
    

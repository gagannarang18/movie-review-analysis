import streamlit as st
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model.
try:
    model_path = os.path.join(os.path.dirname(__file__), 'movie_sentiment.h5')
    model = load_model(model_path, compile=False)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the tokenizer.
try:
    tokenizer_path = os.path.join(os.path.dirname(__file__), 'tokenizer_1.pickle')
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

def predict_sentiment(text, max_length=1422):
    try:
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=max_length)
        prediction = model.predict(padded_sequences, verbose=0)
        sentiment = 'Positive' if prediction[0] > 0.5 else 'Negative'
        confidence = float(prediction[0][0]) if sentiment == 'Positive' else 1 - float(prediction[0][0])
        return sentiment, confidence
    except Exception as e:
        return f"Error: {e}", 0.0

# Streamlit App UI.
st.title("Movie Sentiment Analysis")
st.write("Enter a movie review to analyze its sentiment:")

input_text = st.text_area("Review Text:", "This was a great movie! The acting was superb.")
if st.button("Analyze Sentiment"):
    sentiment, confidence = predict_sentiment(input_text)
    st.subheader("Result:")
    if "Error" not in sentiment:
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2%}")
        if sentiment == "Positive":
            st.success("👍 Positive Review")
        else:
            st.error("👎 Negative Review")
    else:
        st.error(sentiment)

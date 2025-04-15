import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logs

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import pickle

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        color: #FF4B4B;
        text-align: center;
        padding: 1rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        text-align: center;
        font-size: 1.5rem;
    }
    .positive {
        background-color: #D6FFD6;
        color: #228B22;
    }
    .negative {
        background-color: #FFE0E0;
        color: #FF0000;
    }
    .stTextArea textarea {
        min-height: 150px;
    }
</style>
""", unsafe_allow_html=True)

# --- Model and Tokenizer Loading ---
@st.cache_resource
def load_components():
    try:
        model = load_model('movie_review.keras', compile=False)
        with open('movie_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"🚨 Error loading model/tokenizer: {str(e)}")
        st.stop()

# --- Initialize Components ---
model, tokenizer = load_components()
MAX_LENGTH = 1422

# --- Prediction Function ---
def analyze_review(text):
    try:
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAX_LENGTH)
        prediction = model.predict(padded, verbose=0)[0][0]
        return "Positive" if prediction > 0.5 else "Negative"
    except Exception as e:
        return f"Error: {str(e)}"

# --- Streamlit UI ---
st.markdown('<h1 class="header">🎬 Movie Review Sentiment Analyzer</h1>', unsafe_allow_html=True)

with st.container():
    review = st.text_area(
        "Share your movie experience:",
        "The movie was amazing and I totally loved it.",
        help="Write your review and click analyze to get sentiment"
    )

col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("🎯 Analyze Review", use_container_width=True):
        with st.spinner("🍿 Analyzing your review..."):
            sentiment = analyze_review(review)

        if "Error" in sentiment:
            st.error(sentiment)
        else:
            result_style = "positive" if sentiment == "Positive" else "negative"
            st.markdown(f"""
            <div class="result-box {result_style}">
                {sentiment} Sentiment Detected!
                {'🍅' if sentiment == 'Negative' else '🎉'}
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Powered by TensorFlow | Note: AI analysis may not always reflect human perception")
import streamlit as st
import pickle
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Page Configuration
st.set_page_config(page_title="Dual-Engine Sentiment AI", page_icon="🧠", layout="centered")

# 2. Re-introduce Text Preprocessing Functions
def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Safely Load All Assets (Cached to prevent reloading lag)
@st.cache_resource
def load_ml_assets():
    with open("model.pkl", "rb") as m_file:
        ml_model = pickle.load(m_file)
    with open("vectorizer.pkl", "rb") as v_file:
        ml_vectorizer = pickle.load(v_file)
    return ml_model, ml_vectorizer

@st.cache_resource
def load_dl_assets():
    dl_model = tf.keras.models.load_model("lstm_sentiment_model.keras")
    with open("tokenizer.pkl", "rb") as t_file:
        dl_tokenizer = pickle.load(t_file)
    return dl_model, dl_tokenizer

# Initialize asset variants
try:
    ml_model, ml_vectorizer = load_ml_assets()
    dl_model, dl_tokenizer = load_dl_assets()
except Exception as e:
    st.error(f"Critical error loading model files: {e}")
    st.info("Ensure model.pkl, vectorizer.pkl, lstm_sentiment_model.keras, and tokenizer.pkl sit in the same folder.")

# 4. Streamlit UI Layout
st.title("Dual-Engine Sentiment Analysis Dashboard")
st.markdown("Compare predictions between a classic Machine Learning algorithm and a deep Deep Learning LSTM Network running fully offline.")
st.write("---")

# Selection Engine Dropdown
model_choice = st.selectbox(
    "Select Intelligence Engine Layer:",
    [
        "Traditional ML: Logistic Regression (Fast)", 
        "Deep Learning: LSTM Recurrent Neural Network (Context-Aware)"
    ]
)

user_input = st.text_area("Enter textual content for classification:", placeholder="Type your review here...")

if st.button("Run Analytics Pipeline", use_container_width=True):
    if not user_input.strip():
        st.warning("Please input valid text before executing.")
    else:
        cleaned = clean_text(user_input)
        
        # ENGINE 1: LOGISTIC REGRESSION
        if "Traditional ML" in model_choice:
            vectorized_input = ml_vectorizer.transform([cleaned])
            prediction = ml_model.predict(vectorized_input)[0]
            
            if prediction.lower() == 'positive':
                st.success("### Sentiment Result: Positive ")
            else:
                st.error("### Sentiment Result: Negative ")
            st.info("Inference complete via local sparse-matrix calculation loop.")
            
        # ENGINE 2: DEEP LEARNING LSTM 
        elif "Deep Learning" in model_choice:
            # 1. Tokenize string sequence matching training configurations
            seq = dl_tokenizer.texts_to_sequences([cleaned])
            # 2. Pad to the exact sequence length of 200 tokens
            padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
            
            # 3. Predict decimal float probability output (0.0 to 1.0)
            raw_prediction = dl_model.predict(padded)[0][0]
            
            # 4. Apply standard classification threshold boundary
            if raw_prediction >= 0.5:
                st.success(f"### Sentiment Result: Positive ")
                st.caption(f"Network Sentiment Score: **{raw_prediction * 100:.2f}%** confidence toward positive class assignment.")
            else:
                st.error(f"### Sentiment Result: Negative ")
                st.caption(f"Network Sentiment Score: **{(1 - raw_prediction) * 100:.2f}%** confidence toward negative class assignment.")
            st.info("Inference complete via localized Recurrent Neural Network forward pass execution.")
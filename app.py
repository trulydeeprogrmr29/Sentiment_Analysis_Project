import streamlit as st
import pickle
import re

# 1. Page Configuration
st.set_page_config(page_title="Sentiment AI Dashboard", page_icon="*", layout="centered")

# 2. Re-introduce the Text Preprocessing Function
def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Safely Load the Local Model & Vectorizer
@st.cache_resource
def load_assets():
    with open("model.pkl", "rb") as m_file:
        local_model = pickle.load(m_file)
    with open("vectorizer.pkl", "rb") as v_file:
        local_vectorizer = pickle.load(v_file)
    return local_model, local_vectorizer

try:
    model, vectorizer = load_assets()
except FileNotFoundError:
    st.error("Core files missing! Please ensure 'model.pkl' and 'vectorizer.pkl' are inside the project folder.")

# 4. Streamlit UI Layout
st.title("Advanced Sentiment Analysis Dashboard")
st.markdown("Build an end-to-end NLP pipeline using local classification algorithms or cloud-hosted LLM engines.")
st.write("---")

# Model selection toggle
model_choice = st.selectbox(
    "Select Intelligence Engine:",
    [
        "Local Logistic Regression (Fast & Free)", 
        "Anthropic Claude API (Advanced Nuance)", 
        "Google Gemini API (Scalable Reasoning)"
    ]
)

# Text area for user review input
user_input = st.text_area("Enter textual content for classification:", placeholder="Type your movie or product review here...")

if st.button("Run Analytics Pipeline", use_container_width=True):
    if not user_input.strip():
        st.warning("Please input valid text before executing.")
    else:
        # Handle Local Pipeline Execution
        if model_choice == "Local Logistic Regression (Fast & Free)":
            cleaned = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned])
            
            # Predict outcome and confidence
            prediction = model.predict(vectorized_input)[0]
            probabilities = model.predict_proba(vectorized_input)[0]
            
            # Display results beautifully based on sentiment
            if prediction.lower() == 'positive':
                st.success(f"### **Sentiment: Positive **")
            else:
                st.error(f"### **Sentiment: Negative **")
                
            st.info(f"Pipeline metrics processing complete.")
            
        # Placeholders for API Routing Integrations
        elif model_choice == "Anthropic Claude API (Advanced Nuance)":
            st.warning(" Anthropic Router selected. Connect your API token in production to unlock semantic parsing.")
            
        elif model_choice == "Google Gemini API (Scalable Reasoning)":
            st.warning(" Google Gemini Router selected. Connect your Gemini API token to enable real-time cloud inference.")
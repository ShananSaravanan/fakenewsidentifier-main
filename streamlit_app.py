import streamlit as st
import joblib
import requests
from newspaper import Article  # You can install the 'newspaper3k' package for this

# Load the trained model
pipe = joblib.load('model_v1.pkl')

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return "Error extracting text from URL: " + str(e)

# Streamlit app
st.title("Fake News Detection")

# Text input field: Allow user to input either text or a URL
input_text = st.text_area("Enter text or a URL:")

if input_text:
    # Check if the input is a URL (simple check for "http" or "www")
    if input_text.startswith('http') or input_text.startswith('www'):
        st.write("Extracting text from URL...")
        text = extract_text_from_url(input_text)
    else:
        text = input_text

    # Make prediction using the model
    prediction = pipe.predict([text])

    # Output prediction: 0 -> Not Fake, 1 -> Fake
    if prediction[0] == 0:
        st.write("The news is **Not Fake**.")
    else:
        st.write("The news is **Fake**.")

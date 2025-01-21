import streamlit as st
import joblib
import requests
import logging as log
from newspaper import Article  # You can install the 'newspaper3k' package for this
import mlflow
import mlflow.sklearn
import pandas as pd
import os

# running asycnhronous checks
import threading
from data_drift import check_data_drift

def run_async_drift_check(train_data_path, new_data_path):
    def drift_task():
        drift_result = check_data_drift(train_data_path, new_data_path)
        log.info(f"Drift result: {drift_result}")
    # Run drift check in a separate thread
    threading.Thread(target=drift_task).start()


# Initialize MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Replace with your MLflow server URI if applicable
mlflow.set_experiment("Fake News Detection - Runtime Monitoring")

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

    label = prediction[0]

    # Log prediction and input length to MLflow in real-time
    with mlflow.start_run(nested=True):
        mlflow.log_param("input_type", "URL" if input_text.startswith('http') else "Text")
        mlflow.log_metric("input_length", len(text))
        mlflow.log_metric("prediction", prediction)  # Log prediction as a metric (0 or 1)

    

    # checking data drift
    df = pd.DataFrame({
        "processed_text": [text],
        "label": [label]
    })

    output = df.to_excel("output.xlsx", index=False)

    # File to store cumulative data
    cumulative_data_path = "cumulative_data.xlsx"

    def update_cumulative_data(new_data):
        if os.path.exists(cumulative_data_path):
            cumulative_data = pd.read_excel(cumulative_data_path)
            cumulative_data = pd.concat([cumulative_data, new_data], ignore_index=True)
        else:
            cumulative_data = new_data

        cumulative_data.to_excel(cumulative_data_path, index=False)

    # Update cumulative data after every input
    new_entry = pd.DataFrame({"processed_text": [text], "label": [label]})
    update_cumulative_data(new_entry)


    

    # Call the drift detection function
    drift_result = run_async_drift_check("train_data.xlsx", "output.xlsx")

    # Display the result
    #st.write(drift_result)



    # Output prediction: 0 -> Not Fake, 1 -> Fake
    if prediction[0] == 0:
        st.write("The news is **Not Fake**.")
    else:
        st.write("The news is **Fake**.")



# Logs and tracks user interaction
log.basicConfig(level=log.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Log an example message
log.info("Streamlit app started")





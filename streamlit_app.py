import streamlit as st
import joblib
import requests
import logging as log
from newspaper import Article  # You can install the 'newspaper3k' package for this
import mlflow
import mlflow.sklearn
import pandas as pd
import os
from scripts.retraining_model import retrain_model
import glob

# running asycnhronous checks
import threading
from scripts.data_drift import check_data_drift

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
#pipe = joblib.load('model_v1.pkl')
# Function to load the latest model
def load_latest_model(models_dir="models"):
    model_files = glob.glob(f"{models_dir}/model_v*.pkl")
    latest_model = max(model_files, key=os.path.getctime)
    return joblib.load(latest_model)

# Load the most recent model
pipe = load_latest_model()

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

    import time
    start_time = time.time()
    # Make prediction using the model
    prediction = pipe.predict([text])
    end_time = time.time()

    label = prediction[0]


    
    # Log prediction and input length to MLflow in real-time
    with mlflow.start_run(nested=True):
        mlflow.log_param("input_type", "URL" if input_text.startswith('http') else "Text")
        mlflow.log_metric("input_length", len(text))
        mlflow.log_metric("prediction", prediction)  # Log prediction as a metric (0 or 1)
        mlflow.log_metric("execution_time", end_time - start_time)



    

    # checking data drift
    df = pd.DataFrame({
        "processed_text": [text],
        "label": [label]
    })

    output = df.to_excel("data/output.xlsx", index=False)

    # File to store cumulative data
    cumulative_data_path = "data/cumulative_data.xlsx"

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



    # Output prediction: 0 -> Not Fake, 1 -> Fake
    if prediction[0] == 0:
        st.write("The news is **Not Fake**.")
    else:
        st.write("The news is **Fake**.")


if st.button("Check data drift"):

    # Call the drift detection function
    drift_result = check_data_drift("data/train_data.xlsx", "data/cumulative_data.xlsx")
    st.write(drift_result)

    if "Data drift detected" in drift_result:
        st.write("Retraining model due to detected data drift...")
        # Trigger retraining
        new_model_path = retrain_model("data/train_data.xlsx", "data/cumulative_data.xlsx")
        st.write("Model retraining complete.")
    
    

    





# Logs and tracks user interaction
log.basicConfig(level=log.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Log an example message
log.info("Streamlit app started")





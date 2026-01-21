import pandas as pd
from pandas import Timestamp
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import joblib
import os
import re
from datetime import datetime


def retrain_model(train_data_path, cumulative_data_path):

    # Load training dataset
    train_data = pd.read_excel(train_data_path)
    cumulative_data = pd.read_excel(cumulative_data_path)

    # Combine datasets
    updated_data = pd.concat([train_data, cumulative_data], ignore_index=True)

    # Handle missing values in processed_text
    X_train = updated_data['processed_text'].fillna("")  # Features
    y_train = updated_data['label']                     # Labels

    # Load testing dataset
    test_data = pd.read_excel('data/test_data.xlsx')  # Replace with your testing file path

    # Handle missing values in processed_text
    test_data['processed_text'] = test_data['processed_text'].fillna("")  # Replace NaN with an empty string
    X_test = test_data['processed_text']    # Features
    y_test = test_data['label']             # Labels

    # Define pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  # Convert text to numerical features
        ('classifier', MultinomialNB())  # Naive Bayes classifier
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Test the model
    predictions = pipeline.predict(X_test)
    print(classification_report(y_test, predictions))

    # Log retraining with MLflow
    mlflow.set_experiment("Fake News Detection - Retraining")
    with mlflow.start_run() as run:
        # Log parameters, metrics, and the model
        mlflow.log_param("data_size", len(updated_data))
        mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))
        mlflow.log_metric("precision", precision_score(y_test, predictions, zero_division=1))
        mlflow.log_metric("recall", recall_score(y_test, predictions, zero_division=1))
        mlflow.log_metric("f1_score", f1_score(y_test, predictions, zero_division=1))
        mlflow.sklearn.log_model(pipeline, "retrained_model")
        mlflow.log_param("retrain_timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Register the model in MLflow's Model Registry
        model_name = "FakeNewsDetector"
        model_uri = f"runs:/{run.info.run_id}/retrained_model"
        mlflow.register_model(model_uri=model_uri, name=model_name)

        print(f"Model registered with URI: {model_uri}")

    # Save the new model locally
    filename = save_model_with_version(pipeline, directory="models", base_name="model")
    model_path = os.path.join("models", filename)
    joblib.dump(pipeline, model_path)

    return model_path


def save_model_with_version(model, directory="models", base_name="model"):
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)

    # List all files in the directory
    files = os.listdir(directory)
    
    # Regex to match model files with version numbers (e.g., model_v1.pkl, model_v2.pkl)
    pattern = re.compile(rf"{base_name}_v(\d+)\.pkl")
    
    # Find all matching files and extract version numbers
    versions = [int(match.group(1)) for file in files if (match := pattern.match(file))]
    
    # Determine the new version number
    new_version = max(versions, default=0) + 1  # Start from v1 if no models exist
    
    # Construct the new filename
    new_filename = f"{base_name}_v{new_version}.pkl"
    
    # Print the model
    print(f"Model saved as {new_filename}")

    return new_filename

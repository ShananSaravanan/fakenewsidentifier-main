import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import mlflow
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import joblib

# Load training dataset
train_data = pd.read_excel('data/train_data.xlsx')  # Replace with your training file path

# Handle missing values in processed_text
train_data['processed_text'] = train_data['processed_text'].fillna("")  # Replace NaN with an empty string
X_train = train_data['processed_text']  # Features
y_train = train_data['label']           # Labels

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

# mlflow logging during model training & evaluation
mlflow.set_experiment("Fake News Detection - Training")

with mlflow.start_run():
    # Log parameters, metrics, and artifacts
    mlflow.log_param("classifier", "MultinomialNB")
    mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))
    mlflow.log_metric("precision", precision_score(y_test, predictions))
    mlflow.log_metric("recall", recall_score(y_test, predictions))
    mlflow.log_metric("f1_score", f1_score(y_test, predictions))
    mlflow.sklearn.log_model(pipeline, "model")

# Save the trained model to a .pkl file
joblib.dump(pipeline, 'models/model_v1.pkl')
print("Model saved as model_v1.pkl")

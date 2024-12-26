import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Load training dataset
train_data = pd.read_excel('train_data.xlsx')  # Replace with your training file path

# Handle missing values in processed_text
train_data['processed_text'] = train_data['processed_text'].fillna("")  # Replace NaN with an empty string
X_train = train_data['processed_text']  # Features
y_train = train_data['label']           # Labels

# Load testing dataset
test_data = pd.read_excel('test_data.xlsx')  # Replace with your testing file path

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

# Save the trained model to a .pkl file
joblib.dump(pipeline, 'model_v1.pkl')
print("Model saved as model_v1.pkl")

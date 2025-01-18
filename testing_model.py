import joblib
import pandas as pd

# Load the saved model
pipe = joblib.load('model_v1.pkl')

# Example test data: One string of text to predict
test_data = pd.Series(["President Obama shocked the country when he announced he would be running for a third term."])

# Make predictions using the pipeline
predictions = pipe.predict(test_data)

# Output prediction: 0 -> Not Fake, 1 -> Fake
if predictions[0] == 0:
    print("The news is Not Fake.")
else:
    print("The news is Fake.")

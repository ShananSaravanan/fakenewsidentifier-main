from evidently.report import Report
from evidently.metrics import DataDriftTable
from evidently import ColumnMapping
import pandas as pd

def check_data_drift(train_data_path, new_data_path):
    # Load training data
    train_data = pd.read_excel(train_data_path)
    new_data = pd.read_excel(new_data_path)

    # Define column mapping (adjust based on your dataset)
    column_mapping = ColumnMapping(
        # Specify feature and target columns if necessary
        prediction=None,
        target=None,
        id=None,
        numerical_features=[],
        categorical_features=["processed_text","label"]  # Adjust as needed
    )
    
    # Initialize Evidently Report with DataDriftTable metric
    report = Report(metrics=[DataDriftTable()])
    
    # Calculate drift
    report.run(reference_data=train_data, current_data=new_data, column_mapping=column_mapping)
    
    # Save the drift report to an HTML file
    report.save_html("data_drift_report.html")
    
    # Optionally, check for dataset drift and return a message
    drift_metrics = report.as_dict()
    if drift_metrics["metrics"][0]["result"]["dataset_drift"]:
        return "Data drift detected. Retraining recommended."
    else:
        return "No significant data drift detected."

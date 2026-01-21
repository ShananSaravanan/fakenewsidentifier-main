import pandas as pd
import numpy as np
from sklearn import datasets, ensemble

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset

def check_data_drift(train_data_path, new_data_path):

    # Load training data
    train_data = pd.read_excel(train_data_path)
    new_data = pd.read_excel(new_data_path)

    # Skip drift detection for small new data
    if len(new_data) < 500:
        return "Insufficient data for drift detection"

    # Taking samples
    train_data_sample = train_data.sample(n=13000) 

    # Initialize the Evidently report
    # Adjust drift threshold to 10%
    drift_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    drift_report.options.threshold = 0.10  # Adjust threshold to 10%
    
    # Calculate drift
    drift_report.run(reference_data=train_data_sample, current_data=new_data)
    
    # Save the drift report to an HTML file
    drift_report.save_html("data_drift_report.html")
    
    # Optionally, check for dataset drift and return a message
    drift_metrics = drift_report.as_dict()
    drift_score = drift_metrics["metrics"][0]["result"]["dataset_drift"]
    if drift_score:
        print(drift_score)
        if drift_score > 0.1:
            return "Data drift detected"
        else:
            return "No significant data drift detected"
    else:
        return "No data drift detected"

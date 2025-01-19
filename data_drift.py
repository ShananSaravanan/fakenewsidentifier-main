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

    train_data_sample = train_data.sample(n=13000) 

    # Initialize the Evidently report
    drift_report = Report(metrics=[DataDriftPreset()])
    
    # Calculate drift
    drift_report.run(reference_data=train_data_sample, current_data=new_data)
    
    # Save the drift report to an HTML file
    drift_report.save_html("data_drift_report.html")
    
    # Optionally, check for dataset drift and return a message
    drift_metrics = drift_report.as_dict()
    if drift_metrics["metrics"][0]["result"]["dataset_drift"]:
        return "Data drift detected. Retraining recommended."
    else:
        return "No significant data drift detected."

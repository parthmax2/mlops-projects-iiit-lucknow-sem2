# mlflow_run.py

import mlflow
import os
from src.utils import read_yaml
from src.XG_Boost_classifier import load_features_and_labels, train_xgboost_classifier
import numpy as np

# Read config
CONFIG_PATH = "config.yaml"
config = read_yaml(CONFIG_PATH)

# Set MLflow experiment
mlflow.set_experiment("Brain_Tumor_Detection_EfficientNetB0_XGBoost")

with mlflow.start_run():
    # Log model and feature extractor details
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("feature_extractor", "EfficientNetB0")
    mlflow.log_param("n_estimators", config['xgboost']['n_estimators'])
    mlflow.log_param("max_depth", config['xgboost']['max_depth'])
    mlflow.log_param("learning_rate", config['xgboost']['learning_rate'])
    mlflow.log_param("subsample", config['xgboost']['subsample'])

    # Load features and labels
    features, labels = load_features_and_labels(config)

    # Convert one-hot labels to class index to compute accuracy manually
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    test_size = config['base']['test_size']
    random_state = config['base']['random_state']
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Train and save model
    train_xgboost_classifier(features, labels, config)

    # Load model and evaluate
    from joblib import load
    model = load(config['paths']['xgboost_model'])
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Log metrics
    mlflow.log_metric("accuracy", acc)

    # Log classification report as artifact
    report_file = os.path.join(config['paths']['report_dir'], "classification_report.txt")
    if os.path.exists(report_file):
        mlflow.log_artifact(report_file)

    # Log model artifact
    mlflow.log_artifact(config['paths']['xgboost_model'])

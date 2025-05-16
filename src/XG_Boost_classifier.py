import os
import sys
import joblib
import pickle
import numpy as np
import xgboost as xgb
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml

CONFIG_PATH = "config.yaml"

def load_features_and_labels(config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CNN features and one-hot encoded labels from the specified paths.
    """
    try:
        features = np.load(config['paths']['cnn_features_path'])
        labels = np.load(config['paths']['labels_path'])
        logging.info(f"Loaded features shape: {features.shape}, labels shape: {labels.shape}")
        print(f"✅ Features shape: {features.shape}, Labels shape: {labels.shape}")  # <-- [ADDED LINE]
        return features, labels
    except Exception as e:
        logging.error("Error loading CNN features and labels.")
        raise CustomException(e, sys)

def train_xgboost_classifier(X: np.ndarray, y: np.ndarray, config: dict):
    """
    Train an XGBoost classifier on CNN features and save the trained model.
    """
    try:
        # Load model and training config
        params = config['xgboost']
        test_size = config['base']['test_size']
        random_state = config['base']['random_state']

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info("Split data into training and test sets.")
        print("📊 Data split into train and test sets.")  # <-- [ADDED LINE]

        # Convert one-hot encoded labels to class indices
        y_train = np.argmax(y_train, axis=1)  # <-- [ADDED LINE]
        y_test = np.argmax(y_test, axis=1)    # <-- [ADDED LINE]

        # Initialize XGBoost classifier
        clf = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            objective=params['objective'],
            subsample=params['subsample'],
            num_class=params['num_class'],
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

        # Train model
        clf.fit(X_train, y_train)
        logging.info("XGBoost classifier trained successfully.")
        print("✅ XGBoost training complete.")  # <-- [ADDED LINE]

        # Evaluate model
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logging.info(f"Test Accuracy: {acc}")
        logging.info(f"Classification Report:\n{report}")
        print(f"✅ Accuracy: {acc:.4f}")  # <-- [ADDED LINE]

         # Export classification report and accuracy to a text file
        os.makedirs(config['paths']['report_dir'], exist_ok=True)
        report_path = os.path.join(config['paths']['report_dir'], "classification_report.txt")
        with open(report_path, "w") as f:
            f.write("XGBoost Classification Report\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(report)
        logging.info(f"Classification report saved to {report_path}")
        print(f"📄 Report saved to {report_path}")


        # Save trained model
        os.makedirs(config['paths']['model_dir'], exist_ok=True)
        joblib.dump(clf, config['paths']['xgboost_model'])
        logging.info(f"XGBoost model saved to {config['paths']['xgboost_model']}")
        print(f"📁 Model saved to {config['paths']['xgboost_model']}")  # <-- [ADDED LINE]
    except Exception as e:
        logging.error("Error during XGBoost training.")
        raise CustomException(e, sys)
    



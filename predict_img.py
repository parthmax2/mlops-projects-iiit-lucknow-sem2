import os
import sys
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_image
from src.feature_extraction import build_feature_extractor
from tensorflow.keras.applications.efficientnet import preprocess_input
from src.logger import logging
from src.exception import CustomException
from src.config_reader import read_config

def load_models(config):
    try:
        # Load EfficientNet feature extractor
        efficientnet_model = load_model(config['paths']['efficientnet_model_path'])
        logging.info("Loaded EfficientNet feature extractor.")

        # Load trained XGBoost model
        xgboost_model = joblib.load(config['paths']['xgboost_model'])
        logging.info("Loaded trained XGBoost model.")

        # Load label encoder
        label_encoder = joblib.load(config['paths']['label_path'])
        logging.info("Loaded label encoder.")

        return efficientnet_model, xgboost_model, label_encoder

    except Exception as e:
        raise CustomException(e, sys)

def predict_image_class(image_path: str):
    try:
        config = read_config()
        image_size = tuple(config['base']['image_size'])
        channels = config['base']['channels']

        # Preprocess the image
        img = preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 1)

        # Convert grayscale to RGB
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        # Apply EfficientNet preprocessing
        img = preprocess_input(img)

        # Load models
        efficientnet_model, xgboost_model, label_encoder = load_models(config)

        # Extract features using EfficientNet
        features = efficientnet_model.predict(img)
        logging.info(f"Extracted feature shape: {features.shape}")

        # Predict using XGBoost
        pred_class_idx = xgboost_model.predict(features)[0]
        pred_label = label_encoder.inverse_transform([pred_class_idx])[0]
        logging.info(f"Predicted class index: {pred_class_idx}, label: {pred_label}")

        return pred_label

    except Exception as e:
        logging.error("Prediction failed.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Brain Tumor Image Classification")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the brain MRI image")

    args = parser.parse_args()
    prediction = predict_image_class(args.image_path)
    print(f"\n🔍 Predicted Tumor Type: {prediction}")

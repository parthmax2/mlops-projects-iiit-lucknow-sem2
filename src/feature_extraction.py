import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.efficientnet import preprocess_input
import joblib
from src.logger import logging
from src.exception import CustomException
from src.config_reader import read_config
import os

def load_preprocessed_data():
    """
    Load preprocessed features and labels from disk.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Preprocessed features and labels
    """
    try:
        config = read_config()
        features_path = config['paths']['features_path']
        labels_path = config['paths']['labels_path']

        X = np.load(features_path)
        y = np.load(labels_path)

        logging.info(f"Loaded preprocessed data: features {X.shape}, labels {y.shape}")
        return X, y

    except Exception as e:
        logging.error("Failed to load preprocessed data.")
        raise CustomException(e, sys)


def build_feature_extractor(input_shape: tuple) -> Model:
    """
    Load the EfficientNetB0 model as feature extractor.

    Args:
        input_shape (tuple): Shape of the input image.

    Returns:
        Model: Keras model for feature extraction.
    """
    try:
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_tensor=Input(shape=input_shape),
            pooling='avg'
        )

        model = Model(inputs=base_model.input, outputs=base_model.output)
        logging.info("EfficientNetB0 model loaded for feature extraction.")
        return model

    except Exception as e:
        logging.error("Error occurred while building feature extractor.")
        raise CustomException(e, sys)


def extract_cnn_features(X: np.ndarray, model: Model) -> np.ndarray:
    """
    Extract CNN features from preprocessed images.

    Args:
        X (np.ndarray): Preprocessed images.
        model (Model): CNN model for feature extraction.

    Returns:
        np.ndarray: Extracted features.
    """
    try:
        # Convert grayscale to RGB if necessary
        if X.shape[-1] == 1:
            X = np.repeat(X, 3, axis=-1)
            logging.info("Converted grayscale images to RGB.")

        # Preprocess as per EfficientNet requirements
        X = preprocess_input(X)

        features = model.predict(X, verbose=1)
        logging.info(f"Extracted CNN features of shape: {features.shape}")
        return features

    except Exception as e:
        logging.error("Error during CNN feature extraction.")
        raise CustomException(e, sys)


def save_cnn_features(features: np.ndarray, path: str):
    """
    Save extracted features to disk.

    Args:
        features (np.ndarray): CNN features to save.
        path (str): Path to save the features.
    """
    try:
        np.save(path, features)
        logging.info(f"Saved CNN features to {path}")
    except Exception as e:
        logging.error("Failed to save CNN features.")
        raise CustomException(e, sys)


def run_feature_extraction():
    """
    Full pipeline for feature extraction using EfficientNetB0.
    """
    try:
        config = read_config()
        cnn_features_path = config['paths']['cnn_features_path']
        image_size = tuple(config['base']['image_size'])
        channels = config['base']['channels']
        efficientnet_model_path = config['paths']['efficientnet_model_path']

        model = build_feature_extractor(input_shape=(*image_size, channels))

        if os.path.exists(cnn_features_path):
            logging.info("CNN features already exist. Skipping extraction.")
        else:
            X, y = load_preprocessed_data()
            features = extract_cnn_features(X, model)
            save_cnn_features(features, cnn_features_path)

        # Save the EfficientNet feature extractor regardless
        model.save(efficientnet_model_path)
        logging.info(f"EfficientNet model saved to {efficientnet_model_path}")

    except Exception as e:
        logging.error("Feature extraction pipeline failed.")
        raise CustomException(e, sys)

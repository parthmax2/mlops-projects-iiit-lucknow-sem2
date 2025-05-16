import os
import sys
import cv2
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm  # <-- NEW: import for progress bar
from src.logger import logging
from src.exception import CustomException
from src.config_reader import read_config
import joblib

# Load config
config = read_config()
image_size = tuple(config['base']['image_size'])
use_augmentation = config['base']['use_data_augmentation']

# Preprocessing config
do_denoise = config['preprocessing'].get('apply_denoising', True)
do_equalize = config['preprocessing'].get('apply_histogram_equalization', True)
do_smoothing = config['preprocessing'].get('apply_smoothing', True)
do_sharpening = config['preprocessing'].get('apply_sharpening', True)


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Applies computer vision techniques to preprocess an MRI image.

    Args:
        image_path (str): File path of the image.

    Returns:
        np.ndarray: Preprocessed image.
    """
    try:
        # Read image in grayscale (MRI images are effectively grayscale)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Image not found or unreadable: {image_path}")

        # Resize to model input size
        img = cv2.resize(img, image_size)

        # Apply denoising
        if do_denoise:
            img = cv2.fastNlMeansDenoising(img, h=10)

        # Histogram Equalization
        if do_equalize:
            img = cv2.equalizeHist(img)

        # Smoothing (Gaussian Blur)
        if do_smoothing:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        # Sharpening
        if do_sharpening:
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)

        # Normalize to [0, 1] and expand dims for channel
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)  # shape: (224, 224, 1)
        return img

    except Exception as e:
        logging.error(f"Error preprocessing image: {image_path}")
        raise CustomException(e, sys)


def preprocess_dataset(image_paths: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Preprocess the full dataset of images and labels with progress tracking.

    Args:
        image_paths (List[str]): List of image file paths.
        labels (List[str]): Corresponding class labels.

    Returns:
        Tuple: (images array, encoded labels array, fitted LabelEncoder)
    """
    try:
        logging.info("Starting dataset preprocessing...")

        images = []
        for path in tqdm(image_paths, desc="Preprocessing images"):
            img = preprocess_image(path)
            images.append(img)

        X = np.array(images)

        # Encode string labels to integers
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        y_onehot = to_categorical(y_encoded)

        logging.info(f"Dataset preprocessing complete: {X.shape}, Labels shape: {y_onehot.shape}")
        return X, y_onehot, label_encoder

    except Exception as e:
        logging.error("Error occurred while preprocessing dataset.")
        raise CustomException(e, sys)


def save_processed_data(X: np.ndarray, y: np.ndarray, label_encoder: LabelEncoder):
    try:
        np.save(config['paths']['features_path'], X)
        np.save(config['paths']['labels_path'], y)
        joblib.dump(label_encoder, config['paths']['label_path'])
        logging.info("Saved preprocessed data and label encoder.")
    except Exception as e:
        raise CustomException(e, sys)

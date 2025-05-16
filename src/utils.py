import os
import pickle
import matplotlib.pyplot as plt
import yaml
from tensorflow.keras.models import save_model
from src.exception import CustomException


def save_keras_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def plot_training_history(history, path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

def read_yaml(path_to_yaml: str) -> dict:
    """Reads a YAML file and returns its contents as a dictionary."""
    try:
        with open(path_to_yaml, 'r') as file:
            content = yaml.safe_load(file)
            return content
    except Exception as e:
        raise CustomException(f"Error reading YAML file {path_to_yaml}: {str(e)}") from e

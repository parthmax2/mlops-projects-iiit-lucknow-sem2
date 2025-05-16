import os
import numpy as np
import joblib
from src.data_loader import get_image_paths_and_labels
from src.preprocessing import preprocess_dataset, save_processed_data
from src.visualization import visualize_preprocessed_images

if __name__ == "__main__":
    # Updated paths based on your project structure
    features_path = "data/processed/features.npy"
    labels_path = "data/processed/labels.npy"
    label_path = "data/processed/label_encoder.pkl"

    # Optional reprocessing: set to True to ignore cache
    force_reprocess = False

    if (
        not force_reprocess and
        os.path.exists(features_path) and 
        os.path.exists(labels_path) and 
        os.path.exists(label_path)
    ):
        print("Preprocessed data found. Loading from saved files...")
        X = np.load(features_path)
        y = np.load(labels_path)
        label_encoder = joblib.load(label_path)
    else:
        # Process raw images
        image_paths, labels = get_image_paths_and_labels("data/raw/Training")
        print(f"Loaded {len(image_paths)} images and labels.")
        X, y, label_encoder = preprocess_dataset(image_paths, labels)
        save_processed_data(X, y, label_encoder)
        print(f"Saved features: {X.shape}, labels: {y.shape}")


# Visualize sample preprocessed images (optional)
    image_paths, _ = get_image_paths_and_labels("data/raw/Training")  #  add this to reuse image paths
    visualize_preprocessed_images(image_paths, num_images=5)           # show first 16 images.

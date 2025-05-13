import json
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
INPUT_JSON = r"C:\Users\user\OneDrive\Desktop\ML\Mlops\music_genre_prediction_mlops_project_MSA24025\data\data.json"
OUTPUT_DIR = r"C:\Users\user\OneDrive\Desktop\ML\Mlops\music_genre_prediction_mlops_project_MSA24025\data"

def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def preprocess_data(json_path, test_size=0.2, random_state=42):
    X, y = load_data(json_path)
    print(f"Loaded MFCCs with shape: {X.shape}, Labels: {y.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Save to .npz file
    np.savez_compressed(f"{OUTPUT_DIR}/preprocessed_data.npz",
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test)

    print(f"Saved preprocessed data to: {OUTPUT_DIR}/preprocessed_data.npz")

# Run it
if __name__ == "__main__":
    preprocess_data(INPUT_JSON)

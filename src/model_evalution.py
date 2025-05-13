import numpy as np
import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model
import os
from sklearn.metrics import classification_report, confusion_matrix

# ------------------ CONFIG ------------------ #
DATA_PATH = os.path.join("data", "preprocessed_data.npz")
MODEL_PATH = os.path.join("models", "music_genre_cnn_model.h5")
EXPERIMENT_NAME = "music_genre_prediction_eval"
# -------------------------------------------- #

def load_data(data_path):
    data = np.load(data_path)
    X_test = data["X_test"][..., np.newaxis]
    y_test = data["y_test"]
    return X_test, y_test

def main():
    # Start MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        # Load model
        model = load_model(MODEL_PATH)

        # Load test data
        X_test, y_test = load_data(DATA_PATH)

        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
        print(f"Test Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        # Predict
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Log to MLflow
        mlflow.log_param("model_path", MODEL_PATH)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)

        # Optional: log classification report
        report = classification_report(y_test, y_pred_classes, output_dict=True)
        for label, scores in report.items():
            if isinstance(scores, dict):
                for metric_name, score in scores.items():
                    mlflow.log_metric(f"{label}_{metric_name}", score)

        print("Evaluation metrics logged to MLflow.")

if __name__ == "__main__":
    main()

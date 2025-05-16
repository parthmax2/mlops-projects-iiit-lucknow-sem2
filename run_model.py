from src.XG_Boost_classifier import load_features_and_labels, train_xgboost_classifier
from src.utils import read_yaml
from src.logger import logging
from src.exception import CustomException

if __name__ == "__main__":
    try:
        print("Starting XGBoost training pipeline...")  # <-- [ADDED LINE for user feedback]
        config = read_yaml("config.yaml")
        logging.info("Starting XGBoost training pipeline.")

        X, y = load_features_and_labels(config)
        train_xgboost_classifier(X, y, config)

        print("Model training completed and saved.")  # <-- [UPDATED print message]
        logging.info("XGBoost training pipeline completed successfully.")

    except CustomException as e:
        print(f"Pipeline failed: {e}")  # <-- [ADDED LINE to display error in terminal]
        logging.error(f"Pipeline failed: {e}")

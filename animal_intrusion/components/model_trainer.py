import os
import sys
import shutil
import zipfile
from ultralytics import YOLO
from animal_intrusion.logger import logging
from animal_intrusion.exception import AppException
from animal_intrusion.entity.config_entity import ModelTrainerConfig
from animal_intrusion.entity.artifacts_entity import ModelTrainerArtifact


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            # Step 1: Unzip data.zip
            logging.info("Unzipping data.zip into data/ directory")
            os.makedirs("data", exist_ok=True)

            with zipfile.ZipFile("data.zip", 'r') as zip_ref:
                zip_ref.extractall("temp_data")

            for item in os.listdir("temp_data"):
                src_path = os.path.join("temp_data", item)
                dst_path = os.path.join("data", item)

                if os.path.exists(dst_path):
                    logging.warning(f"Skipped moving '{item}' as it already exists in 'data/'")
                    continue

                shutil.move(src_path, dst_path)

            shutil.rmtree("temp_data")
            os.remove("data.zip")

            # Step 2: Train YOLOv8 model
            logging.info("Starting YOLOv8 training")
            model = YOLO(self.model_trainer_config.weight_name)  # e.g., yolov8n.pt

            model.train(
                data=self.model_trainer_config.data_yaml_path,  # Absolute path from config
                imgsz=416,
                batch=self.model_trainer_config.batch_size,
                epochs=self.model_trainer_config.no_epochs,
                name="yolov8_results",
                project="runs/train",
                cache=True
            )

            # Step 3: Save best model
            trained_model_src = "runs/train/yolov8_results/weights/best.pt"
            trained_model_dst = os.path.join(self.model_trainer_config.model_trainer_dir, "best.pt")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)

            shutil.copy(trained_model_src, trained_model_dst)
            shutil.copy(trained_model_src, "best.pt")  # Optional copy to root for later use

            # Step 4: Clean up
            shutil.rmtree("runs/train")
            # Do not delete "data/" as it may contain persistent data.yaml

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=trained_model_dst,
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)

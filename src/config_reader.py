import yaml
import os
import sys
from src.logger import logging
from src.exception import CustomException

def read_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Config loaded from {config_path}")
        return config
    except Exception as e:
        logging.error("Error loading config")
        raise CustomException(e, sys)

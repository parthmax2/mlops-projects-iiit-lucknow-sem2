
import os
import shutil
from src.constants import MODEL_PATH, PRETRAINED_MODEL_PATH, CUSTOM_MODEL_NAME

def copy_pipeline_config():
    os.makedirs(os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME), exist_ok=True)
    shutil.copy(
        os.path.join(PRETRAINED_MODEL_PATH, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config'),
        os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME)
    )

import os
import tensorflow as tf
from object_detection.utils import config_util, label_map_util, visualization_utils as viz_utils
from object_detection.builders import model_builder
from src.constants import CONFIG_PATH, CHECKPOINT_PATH

def load_model():
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-19')).expect_partial()

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    return detect_fn
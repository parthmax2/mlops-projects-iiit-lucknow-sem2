
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from src.constants import CONFIG_PATH, PRETRAINED_MODEL_PATH, PREPROCESSDATA_PATH

def update_config():
    config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

    with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = 5
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH + '/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = PREPROCESSDATA_PATH + '/label_map.pbtxt'
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [PREPROCESSDATA_PATH + '/train.record']
    pipeline_config.eval_input_reader[0].label_map_path = PREPROCESSDATA_PATH + '/label_map.pbtxt'
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [PREPROCESSDATA_PATH + '/test.record']

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
        f.write(config_text)
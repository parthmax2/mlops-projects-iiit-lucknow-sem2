
from src.constants import APIMODEL_PATH, MODEL_PATH, CONFIG_PATH, CUSTOM_MODEL_NAME


def get_training_command():
    return print("""python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps=20000""".format(APIMODEL_PATH, MODEL_PATH,CUSTOM_MODEL_NAME,MODEL_PATH,CUSTOM_MODEL_NAME))
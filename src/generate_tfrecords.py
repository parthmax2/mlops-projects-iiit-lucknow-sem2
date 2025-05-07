
import os
from src.constants import SCRIPTS_PATH, IMAGE_PATH, PREPROCESSDATA_PATH

def generate_tfrecords():
    os.system(f"python {SCRIPTS_PATH}/generate_tfrecord.py -x {IMAGE_PATH}/train -l {PREPROCESSDATA_PATH}/label_map.pbtxt -o {PREPROCESSDATA_PATH}/train.record")
    os.system(f"python {SCRIPTS_PATH}/generate_tfrecord.py -x {IMAGE_PATH}/test -l {PREPROCESSDATA_PATH}/label_map.pbtxt -o {PREPROCESSDATA_PATH}/test.record")


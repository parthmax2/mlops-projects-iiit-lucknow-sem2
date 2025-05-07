
import streamlit as st
import cv2
from src.load_model import load_model
from src.constants import PREPROCESSDATA_PATH
from object_detection.utils import label_map_util, visualization_utils as viz_utils
import numpy as np
import tensorflow as tf

st.set_page_config(layout="wide")
st.title("Sign Language Recognition - Real-Time")

# Load model and label map
detect_fn = load_model()
category_index = label_map_util.create_category_index_from_labelmap(
    PREPROCESSDATA_PATH + '/label_map.pbtxt'
)

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

# OpenCV video capture
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to read from camera.")
        break

    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {k: v[0, :num_detections].numpy() for k, v in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=.5,
        agnostic_mode=False
    )

    FRAME_WINDOW.image(image_np, channels="BGR")

cap.release()

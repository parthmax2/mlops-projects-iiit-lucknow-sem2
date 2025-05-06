import streamlit as st
import os
import cv2
from ultralytics import YOLO
from utils.inference import process_video
import tempfile
from suspicious_activity_detection import detect_suspicious_activities

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Intrusion Detection System",
    layout="centered",
    page_icon="🚨"
)
st.markdown(
    "<h1 style='text-align: center; color: #d11a2a;'>🚨 Intrusion Detection System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Upload surveillance footage to detect intrusions like unauthorized persons, vehicles, drones, or weapons.</p>",
    unsafe_allow_html=True
)

# -------------------- Sidebar: Settings --------------------
with st.sidebar:
    st.header(" Detection Settings")
    confidence_threshold = st.slider("Detection Confidence Threshold", 0.2, 1.0, 0.5, 0.05)
    st.info("Adjust to reduce false positives or detect more confidently.")

# -------------------- Directories Setup --------------------
UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "outputs"
MODEL_PATH = "models/yolov8_model.pt"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- File Uploader --------------------
uploaded_file = st.file_uploader(" Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=UPLOAD_DIR) as temp_file:
        temp_file.write(uploaded_file.read())
        input_path = temp_file.name

    st.success("✅ Video uploaded successfully!")
    st.video(input_path)

    # Run inference on video
    if st.button("▶️ Start Intrusion Detection"):
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model not found at `{MODEL_PATH}`. Please ensure the trained YOLO model is available.")
        else:
            with st.spinner("🚀 Running detection. Please wait..."):
                try:
                    model = YOLO(MODEL_PATH)
                    output_path = os.path.join(OUTPUT_DIR, f"processed_{os.path.basename(input_path)}")

                    # Run detection and get results
                    output_video_path, detections = process_video(
                        input_path=input_path,
                        model=model,
                        confidence_threshold=confidence_threshold,
                        output_path=output_path
                    )

                    st.success("✅ Detection complete! Intrusions (if any) are highlighted.")
                    st.video(output_video_path)

                    with open(output_video_path, "rb") as file:
                        st.download_button(
                            label="⬇️ Download Processed Video",
                            data=file,
                            file_name=os.path.basename(output_video_path),
                            mime="video/mp4"
                        )

                    # ---------------------- FIXED INTEGRATION ----------------------
                    # Construct compatible detection format for suspicious activity module
                    formatted_detections = []

                    for det in detections:
                        detection = {
                            "frame": det.get("frame"),
                            "id": det.get("id"),
                            "class": det.get("class"),
                            "bbox": det.get("bbox"),  # Format: [x1, y1, x2, y2]
                            "pose": det.get("pose_keypoints", {})  # Optional
                        }
                        formatted_detections.append(detection)

                    suspicious_events = detect_suspicious_activities(formatted_detections)

                    if suspicious_events:
                        st.warning("⚠️ Suspicious activity detected!")
                        for event in suspicious_events:
                            st.write(f"- **{event['type']}** detected at frame {event['frame']} (ID: {event['id']})")
                    else:
                        st.success("✅ No suspicious activity detected.")

                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from src.preprocessing import preprocess_image
from src.feature_extraction import build_feature_extractor
from src.config_reader import read_config
from src.logger import logging
from src.exception import CustomException

# Set Streamlit page config
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

# Title
st.title("🧠 Brain Tumor Detection from MRI Scan")
st.write("Upload an MRI scan image and I will predict the tumor type.")

# Load configuration
config = read_config()
image_size = tuple(config['base']['image_size'])
channels = config['base']['channels']


@st.cache_resource
def load_models():
    try:
        # Load models only once and cache
        efficientnet_model = load_model(config['paths']['efficientnet_model_path'])
        xgboost_model = joblib.load(config['paths']['xgboost_model'])
        label_encoder = joblib.load(config['paths']['label_path'])

        return efficientnet_model, xgboost_model, label_encoder
    except Exception as e:
        raise CustomException(e)


def predict(image: np.ndarray) -> str:
    try:
        # Expand dims and preprocess
        img = np.expand_dims(image, axis=0)  # (1, H, W, 1)

        # Convert grayscale to RGB if needed
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        img = preprocess_input(img)

        # Load models
        efficientnet_model, xgboost_model, label_encoder = load_models()

        # Feature extraction
        features = efficientnet_model.predict(img)

        # XGBoost prediction
        pred_idx = xgboost_model.predict(features)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        return pred_label
    except Exception as e:
        raise CustomException(e)


def main():
    uploaded_file = st.file_uploader("📤 Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Show uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            # Save temporarily
            temp_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Preprocess image
            image = preprocess_image(temp_path)

            # Show preprocessed image 
            # Convert numpy array to PIL Image for display
            pre_img_display = Image.fromarray((image.squeeze() * 255).astype(np.uint8))
            st.image(pre_img_display, caption="🧪 Preprocessed Image", use_column_width=False, width=250)

            # Predict
            with st.spinner("🔍 Analyzing..."):
                prediction = predict(image)
            st.success(f" **Predicted Tumor Type:** {prediction}")

        except CustomException as ce:
            st.error("⚠️ Prediction failed. Please check logs.")
            logging.error(ce)
        except Exception as e:
            st.error("Unexpected error occurred.")
            logging.error(e)


if __name__ == "__main__":
    main()

#  Brain-Tumor-Detection-using-Hybrid-DL-Model

A production-ready **Brain Tumor Detection System** that leverages **EfficientNetB0** for deep feature extraction and **XGBoost** for classification. Built using a full **MLOps pipeline**, this project includes modular code, version control (Git + DVC), experiment tracking (MLflow), and a deployable **Streamlit web application**.

---

## Project Overview

This system detects and classifies brain tumors from MRI images into multiple categories with high accuracy. It is designed with **scalability**, **modularity**, and **reproducibility** in mind — ideal for research, diagnostics support, and real-world deployment.

---

##  Features

-  **Pretrained EfficientNetB0** as a feature extractor
-  **XGBoost** classifier for robust prediction
-  **Modular architecture**: data loading, preprocessing, feature extraction, training, prediction
-  **Streamlit App** to run predictions on single images
-  **MLflow** for experiment tracking and reproducibility
-  **DVC** for dataset versioning and model tracking
-  Detailed logging and custom exception handling
-  Displays both **raw** and **preprocessed** image in UI

---

##  Tech Stack

- **Languages**: Python
- **Deep Learning**: TensorFlow / Keras (EfficientNetB0)
- **Machine Learning**: XGBoost
- **MLOps Tools**:
  - MLflow (experiment tracking)
  - DVC (data & model versioning)
  - Streamlit (app UI)
- **Others**: NumPy, OpenCV, PIL, scikit-learn, joblib

---

##  Project Structure

├── src/
│ ├── config_reader.py
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── feature_extraction.py
│ ├── XG_Boost_classifier.py   (model.py)
│ ├── main.py
│ ├── predict_img.py
│ ├── logger.py
│ └── exception.py
├── app.py
├── config/
│ └── config.yaml
├── artifacts/
│ └── (models, encoders, etc.)
├── data/
├── dvc.yaml
├── requirements.txt
├── MLproject
└── README.md


---

Usage
--> Open http://localhost:5000

--> Upload an MRI scan (JPG/PNG)

--> View:

        - Uploaded image

        - Preprocessed version

        - Predicted tumor type

---

CLI Prediction :- 
For single image inference:
python src/predict_img.py --image_path path/to/image.jpg

# Due to an error in the submodule it returns a 404-error therefore, instead of adding the entire repo this is the link for verification 
[Link](https://github.com/MansiDakhale/MRI_SYNTHESIS_PROJECT)

# 🚀 Project Title
TSGAN: Tumor-Aware Synthesis of Contrast-Enhanced MRI Without Contrast Agent

## 👤 Author
- **Name**: Mansi Girdhar Dakhale
- **Enrollment No.**: MSA24027
- **Program**: MSc AI-ML
- **Institute**: IIIT Lucknow
- **Semester**: 2

## 🔗 Links
- 📁 GitHub Repository: [Link](https://github.com/MansiDakhale/MRI_SYNTHESIS_PROJECT.git)
- 📊 Project Presentation: [PPT](https://link-to-ppt.com)

## 🧠 Problem Statement
Breast cancer diagnosis often relies on contrast-enhanced MRI (CeT1), which requires injecting gadolinium-based contrast agents that are costly, time-consuming, and potentially harmful to patients with kidney issues.
This project addresses the need for contrast-free alternatives by developing a deep learning pipeline to synthesize CeT1 MRI scans directly from non-contrast PreT1 scans, preserving diagnostic quality without requiring contrast agents.

To enhance clinical relevance, the model incorporates tumor segmentation guidance, ensuring the synthesized images are both anatomically and pathologically accurate

## 🛠️ Tech Stack
- Programming: Python
- ML Libraries: Scikit-learn, Pytorch, etc.
- MLOps Tools: DVC, Tensorboard, Streamlit

## ⚙️ MLOps Implementation
-  Data versioning with DVC  
-  Experiment tracking via Tensorboard 
-  Deployment using Streamlit 


## 🗂️ Folder Structure (Optional)
Short overview of your project repo structure.
```
MRI_Synthesis_Project/ ├── data/ │ └── Resized_dataset/ ├── models/ │ ├── tsgan_generator.py │ ├── tsgan_discriminator.py │ └── unet3d_segmentation.py ├── utils/ │ ├── dataset_loader.py │ ├── losses.py │ ├── logger.py │ └── visualization.py ├── md_train.py # TSGAN Training script ├── segment_train.py # Tumor segmentation (3D U-Net) ├── generate_mri.py # Inference Script ├── app.py # Streamlit App for real-time inference ├── requirements.txt └── README.md

---

## 🎓 Course Context

These projects are submitted as part of the **MLOps coursework** for the **MSc AI-ML** program at IIIT Lucknow. The focus is on turning ML models into scalable, production-ready solutions using real-world tools.

---

## 🙌 Acknowledgements

* **Course Instructor**: *Mr. Sandeep Srivastava*
* **Institution**: Indian Institute of Information Technology, Lucknow
* **Academic Year**: 2024–25

---

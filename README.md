
---

#  MLOps Projects – MSc AI-ML | IIIT Lucknow | Semester 2

---

## This repository showcases the MLOps course projects developed by MSc AI-ML students of IIIT Lucknow (Semester 2). Each project demonstrates an end-to-end machine learning solution using MLOps best practices like data versioning, experiment tracking, containerization, and reproducibility.

---

## 🎯 Key Objectives

* Build robust ML pipelines using **DVC**
* Track experiments and models using **MLflow**
* Deploy models with **FastAPI** or **Flask**
* Containerize applications using **Docker**
* Maintain reproducibility and version control using **Git**

---

## 📁 Repository Structure

```plaintext
├── README.md
├── projects/
│   ├── student-name.md        # Individual project write-ups
├── assets/                    # Architecture diagrams, screenshots, etc.
└── .github/                   # (Optional) contribution guidelines, workflows
```

---

## 🚀 Project Submission Instructions

🔁 **Each student must:**

1. **Create a new Git branch** named after your enrollment number (e.g., `msa24021`).
2. Add your project markdown file to the `projects/` folder (`projects/yourname.md`).
3. Update the table below with your project details.
4. **Push your branch** to this repository.

| Student Name    | Enrollment No. | Project Title               | Branch Name | PPT Link                       | Summary                                 |
| --------------- | -------------- | --------------------------- | ----------- | ------------------------------ | --------------------------------------- |
| Example Student | msa24021       | Movie Recommendation System | `msa24021`  | [PPT](https://link-to-ppt.com) | Content-based + collaborative filtering |
| ...             | ...            | ...                         | ...         | ...                            | ...                                     |

---

## 📝 Project Documentation Template

Inside your branch, create a file `projects/yourname.md` with the following structure:

```markdown
# 🚀 Project Title

## 👤 Author
- **Name**: Your Full Name
- **Enrollment No.**: msa24XXX
- **Program**: MSc AI-ML
- **Institute**: IIIT Lucknow
- **Semester**: 2

## 🔗 Links
- 📁 GitHub Repository: [Link](https://github.com/your-mlops-repo)
- 📊 Project Presentation: [PPT](https://link-to-ppt.com)

## 🧠 Problem Statement
Briefly describe the real-world problem your project addresses.

## 🛠️ Tech Stack
- Programming: Python
- ML Libraries: Scikit-learn, TensorFlow, etc.
- MLOps Tools: DVC, MLflow, Docker, FastAPI

## ⚙️ MLOps Implementation
- ✔️ Data versioning with DVC  
- ✔️ Experiment tracking via MLflow  
- ✔️ REST API for inference using FastAPI/Flask  
- ✔️ Docker containerization for portability  

## 📸 Screenshots (Optional)
Insert architecture diagrams, API snapshots, UI screens, etc.

## 🗂️ Folder Structure (Optional)
Short overview of your project repo structure.
```

---

## 🎓 Course Context

These projects are submitted as part of the **MLOps coursework** for the **MSc AI-ML** program at IIIT Lucknow. The focus is on turning ML models into scalable, production-ready solutions using real-world tools.

---

## 🙌 Acknowledgements

* **Course Instructor**: *Mr. Sandeep Srivastava*
* **Institution**: Indian Institute of Information Technology, Lucknow
* **Academic Year**: 2024–25

---



# 💳 Credit Card Fraud Detection using Machine Learning

## 🧠 Overview

Credit and debit card fraud is a growing concern in the digital age. In **FY 2023**, over **29,000 cases** were reported in India alone. These frauds cause **huge financial losses** for banks and seriously impact **customer trust**.

This project aims to build a **machine learning system** that can **identify and predict fraudulent transactions** accurately, helping both consumers and financial institutions stay safe.

---

## 🎯 Project Goals

- Detect fraudulent transactions using historical data.
- Apply advanced machine learning models with high accuracy and recall.
- Solve class imbalance using SMOTE.
- Create time-based and amount-based features to capture transaction behavior.
- Evaluate models using precision, recall, F1-score, and ROC-AUC.

---

## 📊 Dataset

- 📁 Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 🧾 Transactions: 284,808
- 🔢 Features: 31
  - `Time`: Seconds from first transaction
  - `Amount`: Transaction amount
  - `Class`: 1 = Fraud, 0 = Non-Fraud
  - `V1–V28`: PCA-transformed confidential features

---

## 🛠️ Feature Engineering

New features were derived from the raw dataset to improve model learning:

### ⏱ Time-Based Features:
- `Hour of Transaction`
- `Transactions in Last Hour`
- `Average Amount per Hour`

### 💰 Amount-Based Features:
- `Log_Amount` – log-transformed to reduce skewness
- `Amount_ZScore` – helps detect unusual amounts

### 📊 Statistical Features:
- `V_Mean`: Mean of V1–V28
- `V_Std`: Standard deviation of V1–V28

---

## ⚖️ Handling Imbalanced Data

Fraudulent transactions were very few in the dataset. To fix this, we used **SMOTE (Synthetic Minority Oversampling Technique)**:

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

This balances the classes by generating synthetic fraud cases.

---

## 🤖 Models Used

We trained multiple models and compared their performance:

| Model               | Highlights                                |
|--------------------|--------------------------------------------|
| Logistic Regression| Fast & interpretable                      |
| Decision Tree       | Rule-based model                          |
| Random Forest       | Ensemble of decision trees                |
| XGBoost             | Gradient boosting, very powerful          |
| Neural Network      | Deep learning model, captures complexity  |

---

## 📈 Model Evaluation Metrics

Each model was evaluated using:

- **Accuracy**
- **Precision** (how many predicted frauds were correct)
- **Recall** (how many actual frauds were detected)
- **F1-Score** (balance between precision and recall)
- **ROC-AUC Curve**
- **Confusion Matrix**

### Example (Logistic Regression):

```
Accuracy  : 0.97
Precision : 0.85
Recall    : 0.91
F1-Score  : 0.88
```

---

## 📉 Confusion Matrix

| Actual \ Predicted | Non-Fraud | Fraud |
|---------------------|-----------|-------|
| Non-Fraud           | TN        | FP    |
| Fraud               | FN        | TP    |

- **TP**: Correctly identified fraud
- **FP**: False alarm
- **FN**: Missed fraud (dangerous)
- **TN**: Correctly identified non-fraud

---

## 🧠 Neural Network Overview

- Input layer: All features (Time, Amount, V1–V28, etc.)
- Hidden layers: Process data to learn patterns
- Output layer: Binary classification (Fraud / Not Fraud)

---

## ✅ Conclusion

This project shows how machine learning can help detect fraudulent transactions with high accuracy. By using **feature engineering**, **SMOTE**, and **advanced models**, we can build systems that are fast, reliable, and useful in real-world banking applications.

---

## 🚀 Future Improvements

- Add Explainability with SHAP or LIME
- Real-time prediction with Flask or FastAPI
- Streamlit dashboard for live monitoring
- Deploy as a microservice with Docker

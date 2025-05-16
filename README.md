

# 🌾 Crop Recommendation System

The **Crop Recommendation System** is a machine learning-based web application that suggests the most suitable crop to grow based on real-time soil and environmental parameters. This tool is designed to assist farmers and agricultural planners in making informed decisions to boost productivity and sustainability.

---

## 🚀 Features

### ✅ Accurate Crop Predictions

Based on the analysis of the following seven key inputs:

* **Nitrogen (N)**
* **Phosphorus (P)**
* **Potassium (K)**
* **pH Level**
* **Temperature**
* **Humidity**
* **Rainfall**

### 🧠 Machine Learning Model

* Algorithm: **Random Forest Classifier**
* Trained using a reliable dataset sourced from [Kaggle](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset)
* High prediction accuracy through preprocessing, feature tuning, and model optimization

### 🌐 Web Interface

* Built with **Flask** for backend functionality
* Simple and user-friendly frontend using **HTML**, **CSS**, and **JavaScript**
* Predict the best crop based on real-time inputs

---

## 💻 Technology Stack

**Frontend:**

* HTML5
* CSS3
* JavaScript

**Backend:**

* Python
* Flask
* Scikit-learn
* Pickle (for model serialization)

---

## 📦 Installation

### Prerequisites

* Python 3.x
* Flask
* pip (Python package installer)

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/jatin1234-pp/crop-recommendation.git
cd crop-recommendation

# (Optional) Create and activate a virtual environment
python -m venv env
source env/bin/activate      # On Windows: .\env\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Run the app
python app.py
```

Once the server is running, open your browser and visit:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧪 Usage

1. Enter the values for Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.
2. Click **"Get Recommendation"**.
3. The system will suggest the most suitable crop to grow under those conditions.


---


## 👤 Author

Developed and maintained by [Jatin Saini](https://github.com/jatin1234-pp)


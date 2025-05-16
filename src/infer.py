import requests
import time
import numpy as np

def generate_random_features():
    return np.random.rand(78).tolist()

def send_to_api(features):
    try:
        response = requests.post("http://127.0.0.1:5001/predict", json={"features": features})
        if response.status_code == 200:
            result = response.json()
            print(f"Detected Class: {result['predicted_class']} | Confidence: {result['confidence']}")
        else:
            print("Prediction error:", response.text)
    except Exception as e:
        print("API call failed:", e)

def run_simulation(interval=5):
    while True:
        features = generate_random_features()
        send_to_api(features)
        time.sleep(interval)

if __name__ == "__main__":
    run_simulation()

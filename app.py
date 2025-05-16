import torch
import torch.nn.functional as F
import mlflow.pytorch
from flask import Flask, request, jsonify, render_template
import os
import requests  # Add this import

# Initialize Flask app
app = Flask(__name__)

# Ensure templates folder exists for HTML
app.template_folder = os.path.join(os.path.dirname(__file__), "templates")

# Path to your MLflow model
model_uri = "mlruns/571955796731090884/cdd0faded6bc474db131e8950c8e1a5d/artifacts/cyber_model"

# Load the model
model = mlflow.pytorch.load_model(model_uri)
model.eval()  # Set model to evaluation mode

# Global variable to store latest result
latest_result = {"predicted_class": "-", "confidence": "-"}

@app.route("/")
def home():
    return render_template("index.html")  # Loads the front-end page

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' key in JSON input"}), 400

    features = data["features"]

    if len(features) != 78:
        return jsonify({"error": f"Expected 78 features, got {len(features)}"}), 400

    try:
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()

        # Send the latest result to the update_result endpoint
        requests.post("http://127.0.0.1:5001/update_result", json={
            "predicted_class": int(predicted_class),
            "confidence": round(confidence, 4)
        })

        return jsonify({
            "predicted_class": int(predicted_class),
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/update_result", methods=["POST"])
def update_result():
    global latest_result
    data = request.get_json()
    latest_result = data
    return jsonify({"status": "updated"}), 200

@app.route("/latest", methods=["GET"])
def get_latest_result():
    return jsonify(latest_result)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

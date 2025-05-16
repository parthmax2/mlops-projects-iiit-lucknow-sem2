from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved models
model = joblib.load('employee_attrition_model.pkl')

@app.route('/')
def home():
    return "Employee Attrition Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Assuming the input is a list of feature values
    prediction = model.predict([np.array(data['features'])])
    result = {'prediction': int(prediction[0])}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

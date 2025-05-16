from flask import Flask, request, render_template, url_for
import numpy as np
import pickle

# Load model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

# List of all crops with details, pointing to images in the 'static' folder
crops = [
    {"name": "Rice", "image": "rice.jpg", "description": "Rice is a staple crop grown primarily in flooded fields. It thrives in hot, humid climates with regular rainfall."},
    {"name": "Maize", "image": "maize.jpg", "description": "Maize (corn) is versatile, used for food, livestock feed, and industrial products, best grown in well-drained soil with moderate rainfall."},
    {"name": "Jute", "image": "jute.jpg", "description": "Jute is a fiber crop used for burlap sacks, ropes, and mats, and it grows best in warm, wet conditions."},
    {"name": "Cotton", "image": "cotton.jpg", "description": "Cotton is a cash crop grown in warm climates; its fibers make textiles, and seeds are processed for oil."},
    {"name": "Coconut", "image": "coconut.jpg", "description": "Coconut is grown in tropical regions for coconuts, oil, and coir, best in sandy soil with sunlight and rainfall."},
    {"name": "Papaya", "image": "papaya.jpg", "description": "Papaya is a tropical fruit thriving in warm climates, rich in vitamins, grown commercially and locally."},
    {"name": "Orange", "image": "orange.jpg", "description": "Oranges are rich in vitamin C, growing best in warm climates with well-drained soil and moderate rainfall."},
    {"name": "Apple", "image": "apple.jpg", "description": "Apple trees grow in temperate climates, require cold winters and well-drained, loamy soil to produce quality fruit."},
    {"name": "Muskmelon", "image": "muskmelon.jpg", "description": "Muskmelon is a sweet fruit needing sunlight, moderate rainfall, and well-drained soil, popular for its refreshing taste."},
    {"name": "Watermelon", "image": "watermelon.jpg", "description": "Watermelon thrives in warm seasons, needing high temperatures and plenty of water, grown for its juicy, sweet fruit."},
    {"name": "Grapes", "image": "grapes.jpg", "description": "Grapes are grown for fresh consumption and wine, needing temperate climates with well-drained soil and moderate rainfall."},
    {"name": "Mango", "image": "mango.jpg", "description": "Mango is a tropical fruit tree, thriving in hot, humid climates, needing well-drained, sandy loam soil with sun exposure."},
    {"name": "Banana", "image": "banana.jpg", "description": "Bananas grow in tropical areas, thriving in hot, humid climates with regular rainfall and well-drained soil."},
    {"name": "Pomegranate", "image": "pomegranate.jpg", "description": "Pomegranate grows best in dry climates, rich in antioxidants, cultivated for fresh consumption and juice production."},
    {"name": "Lentil", "image": "lentil.jpg", "description": "Lentil is a protein-rich legume grown in temperate regions, important for crop rotation due to nitrogen fixation."},
    {"name": "Blackgram", "image": "blackgram.jpg", "description": "Blackgram is a legume crop in tropical areas, an essential protein source in South Asia for curries and dals."},
    {"name": "Mungbean", "image": "mungbean.jpg", "description": "Mungbean is grown for its small, green seeds, thriving in warm climates, commonly used in cooking."},
    {"name": "Mothbeans", "image": "mothbeans.jpg", "description": "Mothbeans are drought-resistant legumes in arid areas, used in curries and soups."},
    {"name": "Pigeonpeas", "image": "pigeonpeas.jpg", "description": "Pigeonpeas are legumes in tropical regions, important for food security due to their high protein content."},
    {"name": "Kidneybeans", "image": "kidneybeans.jpg", "description": "Kidney beans are protein-rich, grown in warm seasons with fertile, well-drained soil."},
    {"name": "Chickpea", "image": "chickpea.jpg", "description": "Chickpea is a pulse crop in semi-arid regions, a major source of plant protein, improving soil health in rotation."},
    {"name": "Coffee", "image": "coffee.jpg", "description": "Coffee is a tropical crop grown in high-altitude areas for its seeds, used in the popular beverage."}
]

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get form input values
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    # Prepare features for prediction
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)[0]  # Predict crop

    # Map prediction result to crop details
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
        6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
        11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
        16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
        20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    # Find crop details
    crop_name = crop_dict.get(prediction, "Unknown Crop")
    crop_info = next((crop for crop in crops if crop['name'] == crop_name), None)

    return render_template('index.html', crop=crop_info)

if __name__ == "__main__":
    app.run(debug=True)
        
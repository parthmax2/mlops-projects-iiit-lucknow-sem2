from flask import Flask, request, render_template_string
import numpy as np
import librosa
import os
import tempfile
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'music_genre_cnn_model.h5'))

model = load_model(MODEL_PATH)

# Genre labels
GENRE_LABELS = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

# Minimal HTML template for upload and result
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Music Genre Classifier 🎵</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 60px; background-color: #f5f5f5; }
        h1 { color: #333; }
        form { margin-top: 30px; }
        input[type=file] { padding: 10px; }
        input[type=submit] {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white; border: none; cursor: pointer;
            border-radius: 5px;
        }
        input[type=submit]:hover { background-color: #45a049; }
        .result { margin-top: 30px; font-size: 1.5em; color: #444; }
    </style>
</head>
<body>
    <h1>🎶 Predict the Music Genre 🎶</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".wav" required><br><br>
        <input type="submit" value="Predict Genre">
    </form>
    {% if genre %}
    <div class="result">🎧 Predicted Genre: <strong>{{ genre }}</strong></div>
    {% endif %}
</body>
</html>
"""

# Extract MFCC features
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=3)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc = mfcc.T  # shape: (n_frames, 13)
        if mfcc.shape[0] < 130:
            pad_width = 130 - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:130]
        mfcc_scaled = mfcc.reshape(1, 130, 13, 1)
        return mfcc_scaled
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template_string(HTML_TEMPLATE, genre="No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template_string(HTML_TEMPLATE, genre="No file selected.")

    # Save to temp file and close it before processing
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    file.save(tmp.name)
    tmp.close()

    features = extract_features(tmp.name)

    # Delete temp file
    try:
        os.unlink(tmp.name)
    except:
        pass

    if features is None:
        return render_template_string(HTML_TEMPLATE, genre="Failed to process file.")

    prediction = model.predict(features)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_genre = GENRE_LABELS[predicted_index]

    return render_template_string(HTML_TEMPLATE, genre=predicted_genre)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, send_from_directory
import os
import cv2
from ultralytics import YOLO
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO('best.pt')  # Replace with your trained model

dangerous_animals = {
    "Bear", "Brown bear", "Bull", "Cheetah", "Crocodile", "Elephant", "Fox",
    "Hippopotamus", "Jaguar", "Kangaroo", "Leopard", "Lion", "Lynx", "Monkey",
    "Otter", "Panda", "Polar bear", "Raccoon", "Rhinoceros", "Scorpion", "Sea lion",
    "Shark", "Snake", "Spider", "Tiger", "Wolf"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.files['video']
        if not video:
            return "No video uploaded", 400

        filename = str(uuid.uuid4()) + ".mp4"
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        video.save(input_path)
        process_video(input_path, output_path)
        return render_template('index.html', output_video=filename)

    return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 0, 255) if class_name in dangerous_animals else (0, 255, 0)
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    app.run(debug=True)

import cv2
from ultralytics import YOLO
from pygame import mixer
import time

# Load the model
model = YOLO(r"C:\Users\dell\Desktop\Projects\Eye_drowziess\Eye_drowsiness\models\last.pt")  # or path to your trained model

# Initialize mixer for alarm
mixer.init()
mixer.music.load(r"C:\Users\dell\Desktop\Projects\Eye_drowziess\Eye_drowsiness\music\music.wav")

# Parameters
closed_eye_threshold = 15  # Number of consecutive frames
closed_eye_counter = 0

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=640, conf=0.5)
    frame_res = frame.copy()

    eye_closed = False

    for r in results:
        for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            class_id = int(cls)
            label = model.names[class_id]

            if label == "closed_eye":
                eye_closed = True
                # Draw bounding box and label
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame_res, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_res, "Closed", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            elif label == "open_eye":
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame_res, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_res, "Open", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Count closed eye frames
    if eye_closed:
        closed_eye_counter += 1
    else:
        closed_eye_counter = 0

    # Trigger alarm
    if closed_eye_counter >= closed_eye_threshold:
        cv2.putText(frame_res, "ALERT! Drowsiness Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)
        if not mixer.music.get_busy():
            mixer.music.play()
    else:
        mixer.music.stop()

    cv2.imshow("Drowsiness Detection", frame_res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

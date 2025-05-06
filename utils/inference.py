from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import os
from tqdm import tqdm

def process_video(input_path, model, confidence_threshold=0.5, output_path="output.mp4"):
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    all_detections = []
    frame_count = 0

    progress_bar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = model(frame, conf=confidence_threshold)[0]
        except Exception as e:
            print(f"Error during model inference on frame {frame_count}: {e}")
            progress_bar.update(1)
            frame_count += 1
            continue

        detections = []

        if hasattr(results, "boxes") and results.boxes is not None:
            for r in results.boxes:
                if r.xyxy is None or r.conf is None or r.cls is None:
                    continue
                x1y1x2y2 = r.xyxy[0]
                if len(x1y1x2y2) < 4:
                    continue

                x1, y1, x2, y2 = map(int, x1y1x2y2)
                if x2 <= x1 or y2 <= y1:
                    continue  # invalid box

                try:
                    conf = float(r.conf[0])
                    cls_id = int(r.cls[0])
                    class_name = model.names[cls_id] if cls_id in model.names else "unknown"
                except Exception as e:
                    print(f"Error extracting detection info: {e}")
                    continue

                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

        try:
            tracks = tracker.update_tracks(detections, frame=frame)
        except Exception as e:
            print(f"Error during tracker update on frame {frame_count}: {e}")
            tracks = []

        current_frame_detections = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            try:
                track_id = track.track_id
                l, t, r, b = track.to_ltrb()
                w, h = r - l, b - t

                detection = track.get_det_class()
                class_name = model.names[detection] if detection is not None and detection in model.names else "unknown"

                cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name} ID: {track_id}', (int(l), int(t) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                current_frame_detections.append({
                    "id": int(track_id),
                    "class": class_name,
                    "bbox": [int(l), int(t), int(w), int(h)]
                })
            except Exception as e:
                print(f"Error processing track: {e}")
                continue

        if current_frame_detections:
            all_detections.append({
                "frame": frame_count,
                "detections": current_frame_detections
            })

        out.write(frame)
        frame_count += 1
        progress_bar.update(1)

    cap.release()
    out.release()
    progress_bar.close()

    return output_path, all_detections

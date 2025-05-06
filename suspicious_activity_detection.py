import numpy as np
from collections import defaultdict, deque
from shapely.geometry import Point, Polygon, LineString

# Persistent storage
object_tracks = defaultdict(lambda: deque(maxlen=30))

# Zones
zone_coords = [(100, 100), (500, 100), (500, 400), (100, 400)]
virtual_line = [(200, 300), (600, 300)]

def is_inside_zone(point, zone=zone_coords):
    return Polygon(zone).contains(Point(point))

def line_crossed(prev, curr, line=virtual_line):
    obj_path = LineString([prev, curr])
    return obj_path.crosses(LineString(line))

def is_arms_flaring(pose_keypoints):
    try:
        left_wrist = pose_keypoints.get('left_wrist')
        right_wrist = pose_keypoints.get('right_wrist')
        left_shoulder = pose_keypoints.get('left_shoulder')
        right_shoulder = pose_keypoints.get('right_shoulder')

        if all([left_wrist, right_wrist, left_shoulder, right_shoulder]):
            left_flared = left_wrist[1] < left_shoulder[1]
            right_flared = right_wrist[1] < right_shoulder[1]
            return left_flared and right_flared
    except:
        return False
    return False



def detect_suspicious_activities(detections):
    suspicious_events = []
    object_tracks = defaultdict(list)  # Local tracking dictionary

    # Track object positions over time
    for det in detections:
        frame = det.get("frame")
        track_id = det.get("id")
        class_name = det.get("class")
        bbox = det.get("bbox")  # (x1, y1, x2, y2)
        pose_keypoints = det.get("pose", {})  # Optional

        if track_id is None or bbox is None:
            continue

        object_tracks[track_id].append({
            "frame": frame,
            "bbox": bbox,
            "class": class_name,
            "pose": pose_keypoints
        })

    # Analyze each tracked object
    for track_id, track in object_tracks.items():
        if len(track) < 2:
            continue

        prev = track[-2]["bbox"]
        curr = track[-1]["bbox"]
        prev_center = ((prev[0] + prev[2]) // 2, (prev[1] + prev[3]) // 2)
        curr_center = ((curr[0] + curr[2]) // 2, (curr[1] + curr[3]) // 2)
        speed = np.linalg.norm(np.array(curr_center) - np.array(prev_center))

        latest_frame = track[-1]["frame"]
        class_name = track[-1]["class"]
        pose_keypoints = track[-1].get("pose", {})

        # Fast movement detection
        if speed > 30 and class_name == "person":
            suspicious_events.append({
                "frame": latest_frame,
                "reason": f"Fast Movement Detected - ID {track_id}"
            })

        # Zone intrusion detection
        if is_inside_zone(curr_center):
            suspicious_events.append({
                "frame": latest_frame,
                "reason": f"Zone Intrusion Detected - ID {track_id}"
            })

        # Virtual line breach
        if line_crossed(prev_center, curr_center):
            suspicious_events.append({
                "frame": latest_frame,
                "reason": f"Virtual Line Breach - ID {track_id}"
            })

        # Weapon detection
        if class_name == "weapon":
            suspicious_events.append({
                "frame": latest_frame,
                "reason": f"Weapon Detected - ID {track_id}"
            })

        # Dropped object detection (stationary for many frames)
        if len(track) >= 10:
            centers = [((t['bbox'][0] + t['bbox'][2]) // 2, (t['bbox'][1] + t['bbox'][3]) // 2) for t in track]
            movement_range = np.ptp(centers, axis=0)
            if all(r < 15 for r in movement_range):
                suspicious_events.append({
                    "frame": latest_frame,
                    "reason": f"Dropped Object Possibly Detected - ID {track_id}"
                })

        # Arms flaring detection using pose estimation
        if class_name == "person" and pose_keypoints:
            if is_arms_flaring(pose_keypoints):
                suspicious_events.append({
                    "frame": latest_frame,
                    "reason": f"Arms Flaring Detected - ID {track_id}"
                })

    # Group Gathering Detection
    centroids = []
    for track in object_tracks.values():
        if len(track) > 0:
            bbox = track[-1]["bbox"]
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            centroids.append(center)

    if len(centroids) >= 3:
        for i in range(len(centroids)):
            group_count = 1
            for j in range(len(centroids)):
                if i != j:
                    distance = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
                    if distance < 100:  # Threshold for "close gathering"
                        group_count += 1
            if group_count >= 3:
                suspicious_events.append({
                    "frame": detections[-1]["frame"],
                    "reason": f"Group Gathering Detected Near ({centroids[i][0]}, {centroids[i][1]})"
                })
                break

    return suspicious_events

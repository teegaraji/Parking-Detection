import json
import pickle

import cv2
import numpy as np
from ultralytics import YOLO

# --- Konfigurasi ---
VIDEO_PATH = "video/input8.mp4"
SLOT_PATH = "anotasi/slot_polygons8.json"
GT_PATH = "ground_truth.pkl"
MODEL_PATH = "models/car_detection.pt"
FRAME_LIMIT = 250  # jumlah frame awal yang dievaluasi

# Load ground truth
with open(GT_PATH, "rb") as f:
    ground_truth = pickle.load(f)

# Load slot polygons
with open(SLOT_PATH) as f:
    slots = json.load(f)

# Load YOLO model
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
correct = 0
total = 0

while frame_idx < FRAME_LIMIT:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi mobil
    results = model(frame)[0]
    detections = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)
            detections.append([x1, y1, x2, y2])

    # Prediksi occupancy slot
    pred_status = []
    for slot in slots:
        pts = np.array(slot["points"], np.int32)
        pts = pts.reshape((-1, 1, 2))
        occupied = False
        for det in detections:
            cx = int((det[0] + det[2]) / 2)
            cy = int((det[1] + det[3]) / 2)
            if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                occupied = True
                break
        pred_status.append(1 if not occupied else 0)  # 1: kosong, 0: terisi

    # Ambil ground truth untuk frame ini
    gt_status = ground_truth.get(frame_idx)
    if gt_status is not None:
        # Bandingkan prediksi dengan ground truth
        for p, g in zip(pred_status, gt_status):
            if p == g:
                correct += 1
            total += 1

    frame_idx += 1

cap.release()

accuracy = correct / total if total > 0 else 0
print(f"Slot occupancy accuracy for {FRAME_LIMIT} frames: {accuracy:.4f}")

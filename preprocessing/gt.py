import json
import pickle

import cv2
import numpy as np

with open("anotasi/slot_polygons8.json") as f:
    slots = json.load(f)

cap = cv2.VideoCapture("video/input8.mp4")
frame_idx = 0
ground_truth = {}

selected_slots = set()
ANNOTATE_INTERVAL = 50


def point_in_poly(pt, poly):
    return (
        cv2.pointPolygonTest(np.array(poly, np.int32).reshape((-1, 1, 2)), pt, False)
        >= 0
    )


def mouse_callback(event, x, y, flags, param):
    global selected_slots
    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, slot in enumerate(slots):
            pts = slot["points"]
            if point_in_poly((x, y), pts):
                if idx in selected_slots:
                    selected_slots.remove(idx)
                else:
                    selected_slots.add(idx)
                break


while True:
    # Set posisi ke frame berikutnya yang akan dianotasi
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    selected_slots = set()
    window_name = "Frame (klik slot kosong, tekan n untuk next, q untuk quit)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        disp = frame.copy()
        for idx, slot in enumerate(slots):
            pts = np.array(slot["points"], np.int32).reshape((-1, 1, 2))
            color = (0, 255, 0) if idx in selected_slots else (0, 0, 255)
            cv2.polylines(disp, [pts], True, color, 2)
            # Draw slot id
            cv2.putText(
                disp,
                str(slot["id"]),
                tuple(pts[0][0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            disp,
            f"Frame {frame_idx} (setiap {ANNOTATE_INTERVAL} frame) - Klik slot kosong, n: next, q: quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("n"):
            break
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            # Save before exit
            with open("ground_truth.pkl", "wb") as f:
                pickle.dump(ground_truth, f)
            print("Keluar dan menyimpan anotasi.")
            exit(0)

    # Simpan status: 1 jika slot kosong (dipilih), 0 jika terisi
    status = [1 if idx in selected_slots else 0 for idx in range(len(slots))]
    # Terapkan status ini untuk frame frame_idx sampai frame_idx+ANNOTATE_INTERVAL-1
    for i in range(frame_idx, frame_idx + ANNOTATE_INTERVAL):
        ground_truth[i] = status.copy()

    frame_idx += ANNOTATE_INTERVAL

cap.release()
cv2.destroyAllWindows()

with open("ground_truth.pkl", "wb") as f:
    pickle.dump(ground_truth, f)
print("Selesai dan menyimpan anotasi.")

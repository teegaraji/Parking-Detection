import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

from cnn_resnet18 import extract_feature_from_bbox


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, feature, kf=None):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1

        self.kf.statePre[:4, 0] = np.array(bbox, dtype=np.float32)
        self.kf.statePre[4:, 0] = 0

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.feature = feature

    def update(self, bbox, feature):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        # Update feature hanya jika feature baru valid
        if feature is not None:
            self.feature = feature
        self.kf.correct(np.array(bbox, dtype=np.float32))
        self.history = []

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.statePost[:4, 0])
        return self.kf.statePost[:4, 0]

    def get_state(self):
        return self.kf.statePost[:4, 0]


def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        - wh
        + 1e-6
    )
    return o


class Tracker:
    def __init__(
        self,
        feature_model,
        max_age=10,
        min_hits=1,
        iou_threshold=0.2,
        cos_threshold=0.8,  # Nilai threshold yang lebih tinggi (dari 0.7 ke 0.8)
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.cos_threshold = cos_threshold
        self.trackers = []
        self.frame_count = 0
        self.feature_model = feature_model
        # Pastikan model dalam mode evaluasi
        self.feature_model.eval()
        # Metrik ID switch
        self.id_switches = 0
        self.prev_frame_objects = {}  # {id: bbox}
        self.object_history = {}  # {id: [position_history]}
        print(f"Tracker initialized with cos_threshold={cos_threshold}")

    def update(self, frame, detections):
        self.frame_count += 1
        print(f"Processing frame {self.frame_count}, detections: {len(detections)}")

        # Predict step untuk semua tracker yang ada
        trks = []
        for t in self.trackers:
            pos = t.predict()
            trks.append(pos)

        # Ekstraksi fitur untuk semua deteksi
        features = []
        valid_dets = []
        h_img, w_img = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det
            # Validasi dan normalisasi bbox
            x1 = max(0, min(w_img - 1, x1))
            y1 = max(0, min(h_img - 1, y1))
            x2 = max(0, min(w_img - 1, x2))
            y2 = max(0, min(h_img - 1, y2))

            # Pastikan bbox valid (lebar dan tinggi > 0)
            if x2 > x1 and y2 > y1:
                # Pertama coba ekstrak fitur dengan bbox original
                feat = extract_feature_from_bbox(
                    frame, [x1, y1, x2, y2], self.feature_model
                )

                # Jika gagal, coba perbesar bbox sedikit
                if feat is None:
                    print(f"[RETRY] Mencoba ekstrak fitur dengan bbox yang lebih besar")
                    pad = 10  # Padding pixels
                    x1_new = max(0, x1 - pad)
                    y1_new = max(0, y1 - pad)
                    x2_new = min(w_img - 1, x2 + pad)
                    y2_new = min(h_img - 1, y2 + pad)
                    feat = extract_feature_from_bbox(
                        frame, [x1_new, y1_new, x2_new, y2_new], self.feature_model
                    )

                # Terakhir, jika masih gagal, gunakan vektor fitur default
                if feat is None:
                    print(
                        f"[WARNING] Gagal ekstrak fitur untuk bbox: {[x1, y1, x2, y2]}"
                    )
                    feat = np.zeros(128, dtype=np.float32)  # Dimensi fitur default

                features.append(feat)
                valid_dets.append([x1, y1, x2, y2])
            else:
                print(f"[WARNING] BBox invalid diabaikan: {[x1, y1, x2, y2]}")

        # Update detections hanya dengan yang valid
        detections = valid_dets

        # Jika tidak ada deteksi valid, kembalikan tracker yang masih aktif
        if len(detections) == 0:
            print("[INFO] Tidak ada deteksi valid pada frame ini")
            self.trackers = [
                t for t in self.trackers if t.time_since_update <= self.max_age
            ]
            results = []
            for t in self.trackers:
                if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                    bbox = t.get_state()
                    bbox = bbox.tolist() if hasattr(bbox, "tolist") else bbox
                    results.append((t.id, bbox))
            print(f"Returning {len(results)} active trackers")
            return results

        # Kasus pertama: belum ada tracker
        if len(trks) == 0:
            print("[INFO] Inisialisasi tracker baru untuk semua deteksi")
            for i, det in enumerate(detections):
                self.trackers.append(KalmanBoxTracker(det, features[i]))

            # Langsung kembalikan hasil tracking
            results = []
            for t in self.trackers:
                bbox = t.get_state()
                bbox = bbox.tolist() if hasattr(bbox, "tolist") else bbox
                results.append((t.id, bbox))
            print(f"Frame pertama: membuat {len(results)} tracker baru")
            return results

        # Matching antara tracker yang ada dengan deteksi baru
        cost_matrix = np.zeros((len(trks), len(detections)), dtype=np.float32)
        for t, tracker in enumerate(self.trackers):
            for d, det in enumerate(detections):
                # Kombinasi IoU dan cosine similarity
                iou_score = iou(trks[t], det)

                # Jika tracker atau deteksi memiliki fitur None, gunakan hanya IoU
                if features[d] is not None and tracker.feature is not None:
                    cos_dist = cosine(tracker.feature, features[d])
                else:
                    cos_dist = (
                        0.5  # Default nilai tengah jika tidak bisa membandingkan fitur
                    )

                # Bobot: 70% IoU, 30% cosine distance
                cost_matrix[t, d] = 0.7 * (1 - iou_score) + 0.3 * cos_dist

        matched, unmatched_trks, unmatched_dets = [], [], []
        if len(trks) > 0 and len(detections) > 0:
            # Gunakan Hungarian algorithm untuk menemukan assignment optimal
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_trks = set()
            assigned_dets = set()

            # Filter assignment berdasarkan threshold
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.cos_threshold:
                    matched.append((r, c))
                    assigned_trks.add(r)
                    assigned_dets.add(c)

            # Identifikasi tracker dan deteksi yang tidak ter-assign
            unmatched_trks = [i for i in range(len(trks)) if i not in assigned_trks]
            unmatched_dets = [
                i for i in range(len(detections)) if i not in assigned_dets
            ]
        else:
            unmatched_trks = list(range(len(trks)))
            unmatched_dets = list(range(len(detections)))

        # Pendekatan baru untuk ID switch
        current_frame_objects = {}

        # Pertama, rekam semua objek yang terdeteksi pada frame ini
        for t, d in matched:
            track_id = self.trackers[t].id
            current_frame_objects[track_id] = detections[d]

            # Jika objek ini baru, tambahkan ke history
            if track_id not in self.object_history:
                self.object_history[track_id] = []

            # Tambahkan posisi saat ini ke history
            self.object_history[track_id].append(detections[d])

        # Cek untuk ID switch - menggunakan trajectory consistency
        for curr_id, curr_bbox in current_frame_objects.items():
            # Cek semua objek dari frame sebelumnya
            for prev_id, prev_bbox in self.prev_frame_objects.items():
                # Jika ID berbeda tapi IoU tinggi dan object history konsisten
                if prev_id != curr_id and iou(prev_bbox, curr_bbox) > 0.6:
                    # Periksa konsistensi trajectory jika history cukup
                    if (
                        prev_id in self.object_history
                        and len(self.object_history[prev_id]) >= 3
                    ):
                        # Hitung velocity konsistensi
                        is_consistent = self._check_trajectory_consistency(
                            self.object_history[prev_id], curr_bbox
                        )

                        if is_consistent:
                            self.id_switches += 1
                            print(
                                f"ID Switch terdeteksi (trajectory): {prev_id} -> {curr_id}"
                            )

        # Update untuk frame berikutnya
        self.prev_frame_objects = current_frame_objects

        # Update tracker yang berhasil di-match
        for t, d in matched:
            self.trackers[t].update(detections[d], features[d])

        # Buat tracker baru untuk deteksi yang tidak ter-match
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i], features[i]))

        # Hapus tracker yang sudah terlalu lama tidak update
        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]

        # Susun hasil tracking
        results = []
        for t in self.trackers:
            # Kembalikan hanya tracker yang sudah mencapai minimum hits
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = t.get_state()
                bbox = bbox.tolist() if hasattr(bbox, "tolist") else bbox
                results.append((t.id, bbox))

        print(
            f"Returning {len(results)} tracks (matched:{len(matched)}, new:{len(unmatched_dets)})"
        )
        # Tambahkan info ID switches ke log
        print(f"Total ID switches: {self.id_switches}")

        return results

    def _check_trajectory_consistency(self, history, current_bbox):
        # Prediksi posisi berdasarkan history
        if len(history) < 2:
            return False

        # Ambil 2 posisi terakhir
        prev1 = history[-1]
        prev2 = history[-2]

        # Hitung predicted velocity
        vx = prev1[0] - prev2[0]
        vy = prev1[1] - prev2[1]

        # Prediksi posisi berikutnya
        pred_x = prev1[0] + vx
        pred_y = prev1[1] + vy

        # Bandingkan dengan posisi aktual
        actual_x = current_bbox[0]
        actual_y = current_bbox[1]

        # Hitung error
        distance_error = ((pred_x - actual_x) ** 2 + (pred_y - actual_y) ** 2) ** 0.5

        # Return True jika error kecil (konsisten)
        return distance_error < 30  # threshold pixel

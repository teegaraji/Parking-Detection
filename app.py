import glob
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
import torch
from ultralytics import YOLO

from cnn_resnet18 import extract_feature_from_bbox, load_embeder_from_pt
from tracking import Tracker


# Load model
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO("models/car_detection.pt")
    detector.to(device)
    embedder = load_embeder_from_pt("models/embedder_triplet.pt").to(device)
    return detector, Tracker(embedder)


detector, tracker = load_model()

# --- Konfigurasi video dari Google Drive ---
# Mapping: video_id -> (drive_file_id, slot_polygon_path, local_filename)
VIDEO_CONFIG = [
    {
        "id": 1,
        "drive_id": "1g452_pqKTBNNoIGE6SnwOThapTiDuvpj",  # Ganti dengan ID Google Drive asli
        "slot_path": "anotasi/slot_polygons1.json",
        "filename": "video/input1.mp4",
    },
    {
        "id": 2,
        "drive_id": "1ztuisKm2nipkJYzPzhDMBSQgRJ8LmbKN",
        "slot_path": "anotasi/slot_polygons2.json",
        "filename": "video/input2.mp4",
    },
    {
        "id": 3,
        "drive_id": "1ziWRU16ikYlLaVq8m9-3I5To85p-LUPj",
        "slot_path": "anotasi/slot_polygons3.json",
        "filename": "video/input3.mp4",
    },
    {
        "id": 4,
        "drive_id": "1-CiiBqf37OPX_L0EVYP57WTeyw3Wdroa",
        "slot_path": "anotasi/slot_polygons4.json",
        "filename": "video/input4.mp4",
    },
    {
        "id": 5,
        "drive_id": "1k1Es8k8ESV1_w3kA4L3dBYB9NBVq0BPP",
        "slot_path": "anotasi/slot_polygons5.json",
        "filename": "video/input5.mp4",
    },
    {
        "id": 6,
        "drive_id": "14djCdke8349btuPnUDig_Nm3uvsFjyHO",
        "slot_path": "anotasi/slot_polygons6.json",
        "filename": "video/input6.mp4",
    },
    {
        "id": 7,
        "drive_id": "1UthosiapTF2C9CWZVGkeXM6c-zbRYUcf",
        "slot_path": "anotasi/slot_polygons7.json",
        "filename": "video/input7.mp4",
    },
    {
        "id": 8,
        "drive_id": "1HIz11DiNss44M-KigpLsf6JgbvrqIoPO",
        "slot_path": "anotasi/slot_polygons8.json",
        "filename": "video/input8.mp4",
    },
    {
        "id": 9,
        "drive_id": "1PUTm9vx0LgVFrpmgSJL85W2AVJXEdm5_",
        "slot_path": "anotasi/slot_polygons9.json",
        "filename": "video/input9.mp4",
    },
    {
        "id": 10,
        "drive_id": "1B62L3YQQXTWGfBpYp03YykUb_HM4kPM_",
        "slot_path": "anotasi/slot_polygons10.json",
        "filename": "video/input10.mp4",
    },
]


def download_from_gdrive(drive_id, dest_path):
    # Download file dari Google Drive (public) menggunakan requests
    # Untuk file besar, gunakan confirm token
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": drive_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token:
        params = {"id": drive_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


# Static config
VIDEO_LIST = [f"video/input{i}.mp4" for i in range(1, 11)]
SLOT_POLYGON_LIST = [f"anotasi/slot_polygons{i}.json" for i in range(1, 11)]

# UI
st.title("Deteksi Slot Parkir Kosong")
video_options = [cfg["id"] for cfg in VIDEO_CONFIG]
video_idx = st.selectbox("Pilih Video Parkiran", video_options)
video_cfg = next(cfg for cfg in VIDEO_CONFIG if cfg["id"] == video_idx)
video_path = video_cfg["filename"]
slot_path = video_cfg["slot_path"]
drive_id = video_cfg["drive_id"]

# Cek apakah video sudah ada, jika belum download dulu
video_file = Path(video_path)
if not video_file.exists():
    # Pastikan folder tujuan ada
    video_file.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner(f"Mengunduh video ID {drive_id} dari Google Drive..."):
        try:
            download_from_gdrive(drive_id, video_path)
            st.success("Video berhasil diunduh.")
        except Exception as e:
            st.error(f"Gagal mengunduh video: {e}")
            st.stop()

# Tampilkan video player
st.video(video_path)

# --- Session State untuk menyimpan hasil deteksi ---
if "slots" not in st.session_state:
    st.session_state.slots = None

# Tambahkan inisialisasi video_process
if "video_process" not in st.session_state:
    st.session_state.video_process = None

cap = cv2.VideoCapture(video_path)
video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
cap.release()
selected_time = st.slider(
    "Pilih waktu (detik) untuk deteksi",
    min_value=0.0,
    max_value=float(video_duration),
    value=float(video_duration) - 1,
    step=1.0,
    format="%.1f",
)

# Tombol untuk mendeteksi kondisi parkir saat ini
if st.button("Cari Parkir Sekarang"):
    # Ambil frame pada waktu yang dipilih user (selected_time)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Hitung frame index dari detik yang dipilih
    frame_idx = int(selected_time * fps)
    # Pastikan frame_idx tidak melebihi jumlah frame
    frame_idx = min(frame_idx, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Gagal mengambil frame dari video.")
    else:
        # Load slot
        with open(slot_path) as f:
            slots = json.load(f)
        st.session_state.slots = slots  # Simpan ke session state

        # Deteksi mobil
        results = detector(frame)[0]
        detections = []
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2])

        # Tracking
        tracks = tracker.update(frame, detections)

        # Buat mapping bbox->track_id
        bbox_id_map = {}
        for track_id, bbox in tracks:
            # Pastikan bbox dalam format list/array 4 elemen
            if bbox is None or len(bbox) != 4:
                continue
            # Cari deteksi dengan IOU tertinggi
            best_iou = 0
            best_det = None
            for det in detections:
                # Hitung IOU
                xx1 = max(bbox[0], det[0])
                yy1 = max(bbox[1], det[1])
                xx2 = min(bbox[2], det[2])
                yy2 = min(bbox[3], det[3])
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h
                area1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                area2 = (det[2] - det[0]) * (det[3] - det[1])
                iou = inter / (area1 + area2 - inter + 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_det = tuple(det)
            if best_det is not None and best_iou > 0.3:
                bbox_id_map[best_det] = track_id

        # Gambar bounding box dan ID (jika ada)
        for det in detections:
            x1, y1, x2, y2 = det
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            if tuple(det) in bbox_id_map:
                track_id = bbox_id_map[tuple(det)]
                # Tampilkan label dan ID mobil di kiri atas bbox
                cv2.putText(
                    frame,
                    f"Mobil ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            else:
                # Jika tidak ada ID, tetap beri label Mobil di kiri atas bbox
                cv2.putText(
                    frame,
                    "Mobil",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # Cek slot kosong
        slot_status = {}
        for slot in slots:
            slot_id = slot["id"]
            pts = np.array(slot["points"], np.int32)
            pts = pts.reshape((-1, 1, 2))
            occupied = False

            # Cek setiap mobil (track) apakah ada di slot ini
            for det in detections:
                cx = int((det[0] + det[2]) / 2)
                cy = int((det[1] + det[3]) / 2)
                if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                    occupied = True
                    break

            color = (0, 0, 255) if occupied else (0, 255, 0)
            slot_status[slot_id] = not occupied  # True jika kosong, False jika terisi
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            # Outline hitam + teks putih
            text_pos = tuple(pts[0][0])
            cv2.putText(
                frame,
                f"ID:{slot_id}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"ID:{slot_id}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        kosong = [str(slot_id) for slot_id, kosong in slot_status.items() if kosong]
        st.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            caption="Hasil Deteksi",
        )

        if kosong:
            st.success(f"Slot kosong ditemukan: {', '.join(kosong)}")
        else:
            st.warning("Tidak ada slot kosong yang terdeteksi.")

# Tombol proses video, hanya aktif jika sudah ada hasil deteksi (slots)
if st.session_state.slots is not None:
    # --- Parameter batch ---
    BATCH_SIZE = 100

    # Inisialisasi batch_num di session state
    if "batch_num" not in st.session_state:
        st.session_state.batch_num = 1

    if (
        st.button("Proses Video Hasil Deteksi")
        or st.session_state.video_process is not None
    ):
        # --- Inisialisasi proses jika belum ada ---
        if st.session_state.video_process is None:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            status_placeholder = st.empty()
            status_placeholder.info("Proses sedang berjalan...")

            # Nama file hasil batch
            batch_num = st.session_state.batch_num
            batch_filename = f"hasil_batch_{batch_num}.mp4"
            batch_filepath = str(Path(tempfile.gettempdir()) / batch_filename)
            out = cv2.VideoWriter(
                batch_filepath,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            progress_bar = st.progress(0)
            preview_placeholder = st.empty()

            frame_idx = 0
            slots = st.session_state.slots

            last_detections = None
            last_results = None

            st.session_state.video_process = {
                "frame_idx": frame_idx,
                "last_detections": last_detections,
                "last_results": last_results,
                "tracker": tracker,
                "batch_filepath": batch_filepath,
                "total_frames": total_frames,
                "slots": slots,
                "video_path": video_path,
                "fps": fps,
                "width": width,
                "height": height,
                "batch_num": batch_num,
            }
            cap.release()
            out.release()

        # --- Lanjutkan proses batch ---
        process = st.session_state.video_process
        frame_idx = process["frame_idx"]
        total_frames = process["total_frames"]
        slots = process["slots"]
        batch_filepath = process["batch_filepath"]
        fps = process["fps"]
        width = process["width"]
        height = process["height"]
        batch_num = process["batch_num"]

        # Buka video dan writer, seek ke frame terakhir
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        out = cv2.VideoWriter(
            batch_filepath,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        progress_bar = st.progress(min(frame_idx / total_frames, 1.0))
        preview_placeholder = st.empty()
        status_placeholder = st.empty()
        status_placeholder.info("Proses batch sedang berjalan...")

        batch_counter = 0
        batch_start_frame = frame_idx
        while frame_idx < total_frames and batch_counter < BATCH_SIZE:
            ret, frame = cap.read()
            if not ret:
                break

            # Proses setiap 2 frame saja untuk preview cepat
            if frame_idx % 2 != 0:
                frame_idx += 1
                continue

            # Deteksi mobil
            results = detector(frame)[0]
            detections = []
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                if int(cls) == 0:
                    x1, y1, x2, y2 = map(int, box)
                    detections.append([x1, y1, x2, y2])

            tracks = tracker.update(frame, detections)
            bbox_id_map = {}
            for track_id, bbox in tracks:
                if bbox is None or len(bbox) != 4:
                    continue
                best_iou = 0
                best_det = None
                for det in detections:
                    xx1 = max(bbox[0], det[0])
                    yy1 = max(bbox[1], det[1])
                    xx2 = min(bbox[2], det[2])
                    yy2 = min(bbox[3], det[3])
                    w = max(0, xx2 - xx1)
                    h = max(0, yy2 - yy1)
                    inter = w * h
                    area1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    area2 = (det[2] - det[0]) * (det[3] - det[1])
                    iou = inter / (area1 + area2 - inter + 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        best_det = tuple(det)
                if best_det is not None and best_iou > 0.3:
                    bbox_id_map[best_det] = track_id

            for det in detections:
                x1, y1, x2, y2 = det
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                if tuple(det) in bbox_id_map:
                    track_id = bbox_id_map[tuple(det)]
                    # Tampilkan label dan ID mobil di kiri atas bbox
                    cv2.putText(
                        frame,
                        f"Mobil ID:{track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "Mobil",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            slot_status = {}
            for slot in slots:
                slot_id = slot["id"]
                pts = np.array(slot["points"], np.int32)
                pts = pts.reshape((-1, 1, 2))
                occupied = False

                for det in detections:
                    cx = int((det[0] + det[2]) / 2)
                    cy = int((det[1] + det[3]) / 2)
                    if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                        occupied = True
                        break

                color = (0, 0, 255) if occupied else (0, 255, 0)
                slot_status[slot_id] = not occupied
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
                text_pos = tuple(pts[0][0])
                cv2.putText(
                    frame,
                    f"ID:{slot_id}",
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"ID:{slot_id}",
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            out.write(frame)

            last_detections = detections
            last_results = results

            frame_idx += 1
            batch_counter += 1
            if frame_idx % 10 == 0 or frame_idx == total_frames:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
            if frame_idx % 50 == 0:
                preview_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    caption=f"Preview Frame {frame_idx}",
                )

        cap.release()
        out.release()
        status_placeholder.empty()

        # Update session state
        st.session_state.video_process.update(
            {
                "frame_idx": frame_idx,
                "last_detections": last_detections,
                "last_results": last_results,
                "batch_filepath": batch_filepath,
                "batch_num": batch_num,
            }
        )

        # Tombol download untuk batch ini (letakkan sebelum logika kelanjutan proses)
        with open(batch_filepath, "rb") as f:
            video_bytes = f.read()
        st.download_button(
            label=f"Unduh Hasil Batch {batch_num} (frame {batch_start_frame+1}-{frame_idx})",
            data=video_bytes,
            file_name=f"hasil_batch_{batch_num}.mp4",
            mime="video/mp4",
            key=f"download_batch_{batch_num}",
        )

        # Jika selesai seluruh video
        if st.session_state.video_process["frame_idx"] >= total_frames:
            st.session_state.video_process = None
            st.success("Proses video selesai!")

            # --- Gabungkan semua batch menjadi satu video ---
            # Cari semua file batch yang sudah dibuat
            temp_dir = tempfile.gettempdir()
            batch_files = sorted(
                glob.glob(str(Path(temp_dir) / "hasil_batch_*.mp4")),
                key=lambda x: int(Path(x).stem.split("_")[-1]),
            )
            if batch_files:
                # Ambil info video dari batch pertama
                cap0 = cv2.VideoCapture(batch_files[0])
                fps = cap0.get(cv2.CAP_PROP_FPS)
                width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap0.release()

                full_video_path = str(Path(temp_dir) / "hasil_full.mp4")
                out_full = cv2.VideoWriter(
                    full_video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

                # Gabungkan semua frame dari setiap batch
                for batch_file in batch_files:
                    cap = cv2.VideoCapture(batch_file)
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out_full.write(frame)
                    cap.release()
                out_full.release()

                # Tampilkan tombol download untuk full video
                with open(full_video_path, "rb") as f:
                    full_bytes = f.read()
                st.download_button(
                    label="Unduh Full Video Gabungan",
                    data=full_bytes,
                    file_name="hasil_full.mp4",
                    mime="video/mp4",
                    key="download_full_video",
                )
        else:
            st.info(
                f"Batch {batch_num} selesai. {st.session_state.video_process['frame_idx']}/{total_frames} frame telah diproses."
            )
            # Tombol lanjutkan proses (hanya tombol ini yang trigger rerun/lanjut)
            if st.button("Lanjutkan Proses", key="continue_process"):
                st.session_state.batch_num += 1
                st.session_state.video_process = None
                st.experimental_rerun()

import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import torch
import cv2
import numpy as np
import av
import threading
import time
from PIL import Image

# ⚠️ WAJIB
st.set_page_config(page_title="Deteksi Drowsy Realtime", layout="centered")

# Fungsi untuk mainkan suara alarm (server-side)
def mainkan_suara_drowsy():
    st.warning("⚠️ Drowsy terdeteksi! Aktifkan suara di browser.")
    st.audio("alarm-restricted-access-355278.mp3", autoplay=True)

# Load model YOLOv5
@st.cache_resource
def load_model():
    model = torch.hub.load(
        'ultralytics/yolov5', 'custom',
        path='best.pt',  # Ganti dengan nama file model kamu yang sudah diunggah
        force_reload=False,
        device='cpu'
    )
    return model

model = load_model()

# UI
st.title("🚨 Deteksi Drowsy (Mengantuk) Realtime")
st.markdown("Webcam akan diakses langsung dari browser kamu. Klik **Start** untuk memulai deteksi.")

# Video processor untuk webrtc
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_sound_time = 0
        self.cooldown = 3
        self.drowsy_active = False
        self.frame_drowsy_tidak_terdeteksi = 0
        self.CONFIDENCE_THRESHOLD = 0.3

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        df = results.pandas().xyxy[0]

        drowsy_detected = False
        current_time = time.time()

        for _, row in df.iterrows():
            if row['confidence'] > self.CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                label = f"{row['name']} {row['confidence']:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if row['name'] == 'drowsy':
                    drowsy_detected = True

        if drowsy_detected:
            self.frame_drowsy_tidak_terdeteksi = 0
            if not self.drowsy_active or (current_time - self.last_sound_time > self.cooldown):
                self.last_sound_time = current_time
                self.drowsy_active = True
                # Tidak bisa memainkan suara dari server, gunakan notifikasi + audio tag
                mainkan_suara_drowsy()
        else:
            self.frame_drowsy_tidak_terdeteksi += 1
            if self.frame_drowsy_tidak_terdeteksi >= 10:
                self.drowsy_active = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Stream dari browser user
webrtc_streamer(
    key="drowsy-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

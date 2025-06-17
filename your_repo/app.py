import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import torch
import numpy as np
import av
import time
import sys
from pathlib import Path
from PIL import Image

# ⚠️ WAJIB: set layout sebelum elemen Streamlit lain
st.set_page_config(page_title="Deteksi Drowsy Realtime", layout="centered")

# Pastikan YOLOv5 path terdeteksi
sys.path.append(str(Path("yolov5")))  # Ini akan aktif karena YOLOv5 diinstal via pip dari GitHub

from models.common import DetectMultiBackend  # dari yolov5

# Fungsi load model
@st.cache_resource
def load_model():
    return DetectMultiBackend(weights="best.pt", device="cpu")

model = load_model()

# Fungsi untuk mainkan alarm drowsy
def mainkan_suara_drowsy():
    st.warning("⚠️ Drowsy terdeteksi! Aktifkan suara di browser.")
    st.audio("alarm-restricted-access-355278.mp3", autoplay=True)

# UI
st.title("🚨 Deteksi Drowsy (Mengantuk) Realtime")
st.markdown("Webcam akan diakses langsung dari browser kamu. Klik **Start** untuk memulai deteksi.")

# Video processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_sound_time = 0
        self.cooldown = 3
        self.drowsy_active = False
        self.frame_drowsy_tidak_terdeteksi = 0
        self.CONFIDENCE_THRESHOLD = 0.3

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, verbose=False)
        df = results.pandas().xyxy[0]

        drowsy_detected = False
        current_time = time.time()

        for _, row in df.iterrows():
            if row['confidence'] > self.CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                label = f"{row['name']} {row['confidence']:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if row['name'].lower() == 'drowsy':
                    drowsy_detected = True

        if drowsy_detected:
            self.frame_drowsy_tidak_terdeteksi = 0
            if not self.drowsy_active or (current_time - self.last_sound_time > self.cooldown):
                self.last_sound_time = current_time
                self.drowsy_active = True
                mainkan_suara_drowsy()
        else:
            self.frame_drowsy_tidak_terdeteksi += 1
            if self.frame_drowsy_tidak_terdeteksi >= 10:
                self.drowsy_active = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start stream
webrtc_streamer(
    key="drowsy-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

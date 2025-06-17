import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import streamlit as st
import av
import torch
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from playsound import playsound
import threading

# ⚠️ WAJIB
st.set_page_config(page_title="Deteksi Drowsy Realtime", layout="centered")

# Fungsi mainkan alarm
def mainkan_suara_drowsy():
    threading.Thread(target=lambda: playsound("alarm-restricted-access-355278.mp3")).start()

# Load model YOLOv5
@st.cache_resource
def load_model():
    model = torch.hub.load(
        'ultralytics/yolov5', 'custom',
        path=r"best.pt",
        force_reload=False,
        device='cpu'
    )
    return model

model = load_model()

# UI
st.title("🚨 Deteksi Drowsy (Mengantuk) Realtime dari Browser")
st.markdown("Webcam akan dibuka di sisi pengunjung. Klik `Start` untuk memulai.")

# Kelas pemroses video
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_sound_time = 0
        self.cooldown = 3
        self.drowsy_active = False
        self.frame_drowsy_tidak_terdeteksi = 0
        self.CONFIDENCE_THRESHOLD = 0.3

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results =

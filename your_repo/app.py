import streamlit as st
import torch
import numpy as np
import av
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from pathlib import Path

# Set halaman
st.set_page_config(page_title="Deteksi Drowsy Realtime", layout="centered")
st.title("😴 Deteksi Drowsy (Ngantuk) Realtime")

# Load model hanya sekali
@st.cache_resource
def load_model():
    return torch.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

model = load_model()

# State untuk kontrol alarm
if "alarm_triggered" not in st.session_state:
    st.session_state.alarm_triggered = False

# Tampilkan audio jika alarm aktif
def show_alarm_audio():
    audio_path = Path("alarm-restricted-access-355278.mp3")
    if audio_path.exists():
        audio_file = open(audio_path, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.error("File alarm.mp3 tidak ditemukan!")

# Kelas pemrosesan video dari kamera
class DrowsyProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.drowsy_detected = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(img, (640, 640))

        results = model(resized)

        # Reset alarm dulu
        self.drowsy_detected = False

        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = map(int, det[:6])
            label = model.names[int(cls)]

            if label.lower() == "drowsy":
                self.drowsy_detected = True

            # Tampilkan bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Jalankan Stream Webcam
ctx = webrtc_streamer(
    key="drowsy-detection",
    video_processor_factory=DrowsyProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Tampilkan alarm jika terdeteksi
if ctx.video_processor:
    if ctx.video_processor.drowsy_detected and not st.session_state.alarm_triggered:
        st.session_state.alarm_triggered = True
        st.warning("⚠️ DROWSY TERDETEKSI!")
        show_alarm_audio()

    elif not ctx.video_processor.drowsy_detected:
        st.session_state.alarm_triggered = False

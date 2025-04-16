import streamlit as st
import pandas as pd
import json
import time
import os
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime

# Konfigurasi Streamlit
st.set_page_config(page_title="ESP32 Monitoring & YOLO Detection", layout="centered")
st.title("🔎 Real-time Monitoring & Human Detection")

# File dan URL
DATA_FILE = 'data.json'
ESP32_CAM_URL = "http://10.200.15.148/capture"  # Ganti IP ESP32-CAM
BUZZER_TRIGGER_URL = "http://10.200.37.215:5000/trigger-buzzer"  # Ganti IP Flask server

st.header("📡 Realtime Sensor Dashboard ESP32")

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w') as f:
        json.dump([], f)

# =================== Aktifkan Buzzer ===================
if st.button("🔊 Bunyi Buzzer"):
    try:
        res = requests.post(BUZZER_TRIGGER_URL, timeout=5)
        if res.status_code == 200:
            st.success("✅ Buzzer berhasil diaktifkan!")
        else:
            st.error("❌ Gagal mengaktifkan buzzer.")
    except requests.exceptions.Timeout:
        st.error("🚨 Waktu habis saat mencoba mengaktifkan buzzer.")
    except Exception as e:
        st.error(f"🚨 Error: {e}")

# Fungsi load data
def load_data():
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Placeholder kontainer
placeholder = st.empty()

with placeholder.container():
    df = load_data()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp', ascending=False).reset_index(drop=True)

        st.subheader("📋 Data Sensor Terbaru")
        st.dataframe(df.head(5), use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌡️ Suhu (°C)", f"{df['temperature'][0]:.1f}")
        with col2:
            st.metric("💧 Kelembaban (%)", f"{df['humidity'][0]:.1f}")
        with col3:
            st.metric("💡 Cahaya", df['light'][0])
        with col4:
            if 'motion' in df.columns:
                motion = df['motion'][0]
                motion_status = "🔴 Terdeteksi" if motion else "⚪ Tidak Ada"
                st.metric("🚶 Gerakan", motion_status)
            else:
                st.metric("🚶 Gerakan", "Data tidak tersedia")

        st.subheader("📈 Grafik Sensor (24 Data Terakhir)")
        chart_df = df.head(24).sort_values(by='timestamp')

        st.line_chart(chart_df[['temperature', 'humidity', 'light']])
        if 'motion' in df.columns:
            st.area_chart(chart_df[['motion']])

# =================== BAGIAN 1: YOLO DETECTION ===================
st.header("🧍 Deteksi Orang via ESP32-CAM (YOLOv3-tiny)")

status_box = st.empty()
stframe = st.empty()
stop = st.button("Stop Webcam")
capture_button = st.button("Manual Capture")
enable_auto_capture = st.checkbox("Aktifkan Auto Capture", value=True)

with st.container():
    st.subheader("📸 Hasil Manual Capture")
    manual_capture_box = st.empty()

with st.container():
    st.subheader("🤖 Hasil Auto Capture")
    auto_capture_box = st.empty()

# Load YOLO
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Fungsi ambil frame dari ESP32-CAM
def get_esp32_frame():
    try:
        response = requests.get(ESP32_CAM_URL, timeout=5)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        st.warning(f"⚠️ Gagal mengambil frame dari ESP32-CAM: {e}")
        return None

running = True
captured = False
auto_captured = False

def update_status(detected):
    if detected:
        status_box.error("🚨 Orang terdeteksi!")
    else:
        status_box.success("✅ Aman, tidak ada orang.")

while running:
    if stop:
        break

    frame = get_esp32_frame()
    if frame is None:
        continue

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences = [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if classes[class_id] == "person" and confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected = len(indexes) > 0
    update_status(detected)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = f"Person: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    stframe.image(frame, channels='BGR', use_container_width=True)

    if capture_button and not captured:
        captured_image = frame.copy()
        manual_capture_box.image(captured_image, channels='BGR', use_container_width=True)
        captured = True
    if not capture_button:
        captured = False

    if enable_auto_capture and detected and not auto_captured:
        max_conf_idx = np.argmax(confidences)
        temp_frame = frame.copy()
        x, y, w, h = boxes[max_conf_idx]
        cv2.rectangle(temp_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        auto_capture_box.image(temp_frame, channels='BGR', use_container_width=True)
        auto_captured = True

    if not detected:
        auto_captured = False

# =================== PEMBATAS ===================
st.markdown("---")

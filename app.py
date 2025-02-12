import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import tempfile

# Load the saved YOLO model
model = YOLO("weights/last.pt")

# Streamlit app layout
st.set_page_config(layout="wide")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("Face Mask Detection App 😷")

# Video Upload Section
st.subheader("Upload a video")
uploaded_file = st.file_uploader("", type=["mp4", "avi", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        res = model(frame)
        output_frame = res[0].plot()
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(output_frame, channels="RGB", use_column_width=True)
    
    cap.release()

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import tempfile

# Load YOLO model
model = YOLO("weights/last.pt")

# Streamlit App Layout
st.set_page_config(layout="wide")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("Face Mask Detection App 😷")

SAMPLE_IMAGES_DIR = "sample image"

def process_and_display(image):
    res = model(image)
    output = res[0].plot()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    col4, col5 = st.columns([1, 1])
    with col4:
        st.success("Output Image")
        st.image(output, caption="Processed Image")
    with col5:
        st.error("Original Image")
        st.image(image, caption="Uploaded Image")

# Image Upload
st.subheader("Upload your image")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    process_and_display(image)

# Sample Images
st.subheader("Try Sample Images")
sample_images = [f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.endswith(('jpg', 'jpeg', 'png'))]
cols = st.columns(4)
for i, image_name in enumerate(sample_images):
    if cols[i % 4].button(f"Sample {i+1}"):
        image = Image.open(os.path.join(SAMPLE_IMAGES_DIR, image_name))
        process_and_display(image)

# Video Upload & Processing
st.subheader("Upload your video")
video_file = st.file_uploader("", type=["mp4", "avi", "mov"])
if video_file:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video.write(video_file.read())
    cap = cv2.VideoCapture(temp_video.name)
    
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        processed_frame = results[0].plot()
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        if out is None:
            h, w, _ = processed_frame.shape
            out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (w, h))
        out.write(processed_frame)
    cap.release()
    out.release()
    st.video(output_video_path)

# Webcam Detection
st.subheader("Real-time Detection")
run_webcam = st.checkbox("Enable Webcam")
if run_webcam:
    camera = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])
    while run_webcam:
        success, frame = camera.read()
        if not success:
            break
        results = model(frame)
        frame = results[0].plot()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    camera.release()

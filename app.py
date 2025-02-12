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

# Path for Sample Images
SAMPLE_IMAGES_DIR = "sample image"

# Image Upload Section
st.subheader("Upload an image or a video")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file:
    file_type = uploaded_file.type
    
    if "image" in file_type:
        image = Image.open(uploaded_file)
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

    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        st.video(video_path)
        
        cap = cv2.VideoCapture(video_path)
        frame_placeholder = st.empty()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            res = model(frame)
            output_frame = res[0].plot()
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(output_frame, channels="RGB")
        
        cap.release()

# Sample Image Section
if not uploaded_file:
    st.subheader("Try Sample Images")
    sample_images = [f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.endswith(('jpg', 'jpeg', 'png'))]

    for i in range(0, len(sample_images), 4):
        cols = st.columns(4)
        for j, image_name in enumerate(sample_images[i:i + 4]):
            if cols[j].button(f"Sample Image {i + j + 1}"):
                uploaded_file = os.path.join(SAMPLE_IMAGES_DIR, image_name)
                image = Image.open(uploaded_file)
                res = model(image)
                output = res[0].plot()
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                col4, col5 = st.columns([1, 1])
                with col4:
                    st.success("Output Image")
                    st.image(output, caption="Output Image")
                with col5:
                    st.error("Original Image")
                    st.image(image, caption="Sample Image")

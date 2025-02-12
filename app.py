import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import tempfile

# Loading the saved model
model = YOLO("weights/last.pt")

# Streamlit app layout
st.set_page_config(layout="wide")

# Create columns for centering the title
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("Face Mask Detection App 😷")

# Upload Image or Video
st.subheader("Upload an Image or Video")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file:
    file_type = uploaded_file.type.split('/')[0]  # Check if it's an image or video
    
    if file_type == "image":
        image = Image.open(uploaded_file)
        res = model(image)
        output = res[0].plot()
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        
        st.image(output, caption="Detected Image")
    
    elif file_type == "video":
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(temp_video_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(temp_video_path)
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame)
            detected_frame = results[0].plot()
            detected_frame = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            
            stframe.image(detected_frame, channels="RGB")
        
        cap.release()
        os.remove(temp_video_path)

st.subheader("Try Sample Images")
SAMPLE_IMAGES_DIR = "sample image"

sample_images = [f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.endswith(("jpg", "jpeg", "png"))]
for i in range(0, len(sample_images), 4):
    cols = st.columns(4)
    for j, image_name in enumerate(sample_images[i:i + 4]):
        if cols[j].button(f"Sample Image {i + j + 1}"):
            image = Image.open(os.path.join(SAMPLE_IMAGES_DIR, image_name))
            res = model(image)
            output = res[0].plot()
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            st.image(output, caption="Detected Sample Image")

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# Load the trained YOLO model
model = YOLO("weights/best.pt")

# Set Streamlit page layout
st.set_page_config(layout="wide")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("Face Mask Detection App 😷")

# Directory for sample images
SAMPLE_IMAGES_DIR = "sample image"

# User Image Upload
st.subheader("Upload your Image for Detection")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# Function to process and display images
def process_and_display(image):
    res = model(image)  # Run YOLO model
    output = res[0].plot()  # Annotate results
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    col4, col5 = st.columns([1, 1])
    with col4:
        st.success("Output Image")
        st.image(output, caption="Detection Result")
    with col5:
        st.error("Original Image")
        st.image(image, caption="Uploaded Image")

# Process Uploaded Image
if uploaded_file:
    image = Image.open(uploaded_file)
    process_and_display(image)

# Sample Images Section
if not uploaded_file:
    st.subheader("Try Sample Images")
    sample_images = [f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.endswith(('jpg', 'jpeg', 'png'))]
    
    for i in range(0, len(sample_images), 4):
        cols = st.columns(4)
        for j, img_name in enumerate(sample_images[i:i+4]):
            if cols[j].button(f"Sample {i + j + 1}"):
                image = Image.open(os.path.join(SAMPLE_IMAGES_DIR, img_name))
                process_and_display(image)

# Webcam Detection
st.subheader("Real-time Face Mask Detection")
run_webcam = st.checkbox("Activate Webcam")
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run_webcam:
    success, frame = camera.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        rgb_annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(rgb_annotated_frame)
    else:
        st.write("Error: Could not access the webcam.")
        break
else:
    camera.release()

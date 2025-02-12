import cv2
import streamlit as st
import tempfile
import ffmpeg
from ultralytics import YOLO
import numpy as np
import os

st.set_page_config(layout="wide")

# Load YOLO Model
model = YOLO("weights/last.pt")

# Video Upload
st.title("Face Mask Detection in Video 🎥")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    
    # Open Video
    cap = cv2.VideoCapture(temp_video.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # FFmpeg Streaming Setup
    output_file = "output.m3u8"
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height))
        .output(output_file, preset="ultrafast", vcodec="h264_nvenc", hls_time=10, hls_list_size=10, f='hls')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    st.subheader("Processed Video 🎞️")
    
    # Process Video Frame-by-Frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for Speed
        frame = cv2.resize(frame, (640, 480))
        
        # Run YOLO detection
        results = model(frame)
        annotated_frame = results[0].plot()

        # Stream Processed Video
        process.stdin.write(annotated_frame.tobytes())

    cap.release()
    process.stdin.close()
    process.wait()

    # Play Streamed Video
    st_player(output_file)

    # Cleanup Temp File
    os.remove(temp_video.name)

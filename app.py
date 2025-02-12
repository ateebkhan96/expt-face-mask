# Importing the Libraries
import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# Loading the saved model
model = YOLO("weights/last.pt")

# # Streamlit app layout
st.set_page_config(layout="wide")

# create 3 columns for centering the text
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.title("Face Mask Detection App 😷")

# Path for Sample Images
SAMPLE_IMAGES_DIR = "sample image"

# Taking the Image from User
st.subheader("Upload your image")
uploaded_file = st.file_uploader("",type=["jpg", "jpeg", "png"])

# If user uploaded an Image
if uploaded_file:
    image = Image.open(uploaded_file)  # Take the image using PIL Library
    res = model(image)                  # Running and Storing the Detection
    output = res[0].plot()              # Getting the output image
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)    # Color correcting the image

    col4, col5 = st.columns([1,1])      # For layout purpose
    with col4:
        st.success("Output Image")
        st.image(output, caption="Output Image")    # Displaying the original image
    with col5:
        st.error("Original Image")
        st.image(image, caption="Uploaded Image",)    # Displaying the original image

    st.subheader("Try Sample Images")
# Display Sample Image Buttons at the bottom if no image is uploaded
if uploaded_file is None:
 st.subheader("Try Sample Images")        # Prompting user to try sample images


# Create columns for sample image buttons, adjusting based on the number of images
sample_images = [f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.endswith(('jpg', 'jpeg', 'png'))]

# Display sample image buttons in rows of 3 columns each
for i in range(0, len(sample_images), 4):
    cols = st.columns(4)
    for j, image_name in enumerate(sample_images[i:i + 4]):
        if cols[j].button(f"Sample Image {i + j + 1}"):
            # Set the selected sample image as `uploaded_file` to reuse processing code
            uploaded_file = os.path.join(SAMPLE_IMAGES_DIR, image_name)

            # Process and display the selected sample image
            image = Image.open(uploaded_file)
            res = model(image)
            output = res[0].plot()
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            # Display the selected sample image and its detection output side by side
            col4, col5 = st.columns([1, 1])
            with col4:
                st.success("Output Image")
                st.image(output, caption="Output Image")
            with col5:
                st.error("Original Image")
                st.image(image, caption="Sample Image")




# # Detecting from image (local machine only)
#
# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])
#
# # Initialize the webcam
# camera = cv2.VideoCapture(0)
#
#
# while run:
#     # Capture frame-by-frame
#     success, frame = camera.read()
#     if success:
#         # Run YOLO inference on the frame
#         results = model(frame)  # Replace 'model' with your actual YOLO model variable
#
#         # Get annotated frame
#         annotated_frame = results[0].plot()  # Process the results to get the visualized frame
#
#         # Convert to RGB format for Streamlit
#         rgb_annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#
#         # Display the frame in Streamlit
#         FRAME_WINDOW.image(rgb_annotated_frame)
#     else:
#         st.write("Error: Could not access the webcam.")
#         break
# else:
#     camera.release()

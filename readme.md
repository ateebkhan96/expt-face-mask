
# 🎭 Face Mask Detection App 😷

This application uses YOLO11 to detect face masks in uploaded or sample images 🎉

# 🌐 Try it Online

Link: [Face-Mask-Detection](https://face-mask-det.streamlit.app/)

# 🚀 Features

* *Mask Detection*: Upload an image, and the app detects if a mask is being worn.
* *Sample Images*: No image? No problem! Try out detection on built-in sample images.
*  *Streamlit-Powered*: Runs as an interactive web app on [Streamlit](https://face-mask-det.streamlit.app/).
* *Run on Local Machine* : Try detection on live webcam
* *Model Training Notebook* : Customize or retrain the model using the included Jupyter notebook.

# 📸 Sample Output
| Output Image                                                     | Uploaded Image                                                      | 
|------------------------------------------------------------------|---------------------------------------------------------------------|
| <img src="Output Sample/with_mask.jpg" width="300" height="400"> | <img src="Output Sample/without_mask.jpg" width="300" height="400"> |


# 📂 Dataset
The model was trained using a face mask dataset sourced from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) uploaded by [Andrew Maranhão](https://www.kaggle.com/andrewmvd), which contains images with and without face masks to enable accurate mask detection.
# 🛠️ Installation On Local Machine

Follow these steps to get the app running on your local machine:

#### 1. Clone the Repo
```bash
git clonehttps://github.com/ateebkhan96/Face-Mask-Detection.git
cd face-mask-detection
```
#### 2. Install Dependecies
```bash
pip install -r requirements.txt
```
#### 3. Run the App:
``` bash
streamlit run app.py
```
    
# 📓 Model Training Notebook

The project includes a Jupyter Notebook ([Yolov11_Face_Mask.ipynb](https://github.com/ateebkhan96/Face-Mask-Detection/blob/main/Yolov11_Face_Mask.ipynb)) to train the YOLO11 model on a custom dataset. You can open this notebook to:

    1. Understand the model training process.
    2. Experiment with hyperparameters and retrain the model.
    3. Generate a new model file (last.pt) for different dataset needs. 

`Note: Make sure to configure the dataset paths within the notebook before training.`

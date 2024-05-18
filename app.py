import os

# Install OpenCV if not already installed
streamlit_logger = os.popen('pip show opencv-python-headless').read()
if "Version" not in streamlit_logger:
    os.system("pip install opencv-python-headless")

# Now you can import OpenCV
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'path_to_your_model.h5'
model = load_model(model_path)

# Define your Streamlit app
def main():
    st.title('Number Recognition')

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_image is not None:
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
        resized_image = cv2.resize(image, (28, 28))
        normalized_image = resized_image.astype('float32') / 255.0
        reshaped_image = np.expand_dims(normalized_image, axis=0)
        
        prediction = model.predict(reshaped_image)
        predicted_label = np.argmax(prediction)

        st.image(image, caption=f"Predicted Label: {predicted_label}", use_column_width=True)

# Run the Streamlit app
if __name__ == '__main__':
    main()


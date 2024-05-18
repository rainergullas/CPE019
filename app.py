import os
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import streamlit as st

st.markdown(
    """
    <style>
    body {
        background-image: url('aurora.jpg'); /* Set background image URL */
        background-size: cover; /* Cover the entire background */
        background-repeat: no-repeat; /* Do not repeat the background image */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    model_path = 'number_detection_model.h5'
    model = load_model(model_path)
    return model

model = load_trained_model()

# Define a function to preprocess input image
def preprocess_image(image):
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28
    resized_image = cv2.resize(grayscale_image, (28, 28))
    # Normalize
    normalized_image = resized_image.astype('float32') / 255.0
    # Reshape to match model input shape
    reshaped_image = np.expand_dims(normalized_image, axis=0)
    return reshaped_image

# Define the Streamlit app
def main():
    st.title('Emtech2 Final Project Number Recognition')

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_image is not None:
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess image
        preprocessed_image = preprocess_image(image)

        # Print preprocessed image shape for debugging
        st.write(f'Preprocessed Image Shape: {preprocessed_image.shape}')

        try:
            # Make prediction
            prediction = model.predict(preprocessed_image)
            predicted_label = np.argmax(prediction)
            st.write(f'Predicted Label: {predicted_label}')
        except Exception as e:
            st.error(f'Error during prediction: {e}')

if __name__ == '__main__':
    main()

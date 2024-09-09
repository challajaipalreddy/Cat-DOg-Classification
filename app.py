import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image, ImageOps
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model




model_path = './models/catdogmodel.h5'
model = load_model(model_path)


# Function to classify faces
def classify_face(image):

    resize = tf.image.resize(image, (256,256))

    yhat = model.predict(np.expand_dims(resize/255, 0))

    if yhat > 0.5: 
        return "dog"
    else:
        return "cat"


# Streamlit app
def main():
    st.title("Cat dog Classifier")
    # st.sidebar.title("Settings")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Convert PIL Image to NumPy array
        image_np = np.array(image)

        # Perform face classification
        result = classify_face(image_np)

        # Display the result
        st.write("Predicted class is:", result)

# Run the appbcdc
if __name__ == "__main__":
    main()

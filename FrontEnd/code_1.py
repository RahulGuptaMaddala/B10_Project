import tensorflow as tf
import cv2
import streamlit as st
import numpy as np
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model("Model.keras")

# Function to add background from URL
def add_bg_from_url():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://www.pexels.com/photo/defocused-image-of-lights-255379/");
            background-attachment: fixed;
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to display the image
def display_img(image):
    try:
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError("Could not read the image")
        img = cv2.resize(img, (800, 600))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img)
        return img
    except Exception as e:
        st.error(f"Error: {e}")

# Add background
add_bg_from_url()

# Load image file
st.title("CLASSIFICATION OF AI AND REAL IMAGES USING DEEP LEARNING")
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

# Perform prediction when button is clicked
if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file)
    
    # Generate prediction button
    if st.button("Generate Prediction"):
        img = display_img(uploaded_file)
        
        if img is not None:
            # Resize image
            resized_img = tf.image.resize(np.array(img), (32, 32))
            
            # Make prediction
            y_pred = model.predict(np.expand_dims(resized_img / 255, 0))
            
            # Display prediction
            if y_pred > 0.5:
                st.markdown("<span style='color:yellow;font-size:42px;'>The given image is REAL </span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:green;font-size:42px;'>The given image is FAKE </span>", unsafe_allow_html=True)

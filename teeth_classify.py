import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Memory optimization (if using GPU)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Display header
st.markdown("<h2 style='color:blue", unsafe_allow_html=True)

# Load the pre-trained model
model = load_model("teeth_classification_model.h5")
class_names = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]

# Display the title
st.markdown("<h1 style='text-align: center;'>Teeth Classification 🦷</h1>", unsafe_allow_html=True)

# Center the file upload box
col1, col2, col3 = st.columns([1, 2, 1])  
with col2:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

# If an image is uploaded
if uploaded_file:
    # Display a spinner while processing
    with st.spinner("Classifying... 🧐"):
        try:
            # Open and preprocess the uploaded image
            img = Image.open(uploaded_file)
            img = img.resize((256, 256))  # Resize to expected input size
            img_array = image.img_to_array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Expand dims and cast to float32

            # Make Prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)

            # Show the result
            st.success(f"Prediction: **{class_names[predicted_class]}**")

        except Exception as e:
            st.error(f"Error: {e}")


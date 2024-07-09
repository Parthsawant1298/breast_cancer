import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# Load the trained model
cnn = tf.keras.models.load_model("breastcancer.keras")

# Create the Streamlit app
st.title("Breast Cancer Detection")
st.write("Upload an image to predict if there is breast cancer.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load image data from file
    img = Image.open(uploaded_file).convert('RGB')
    
    # Display the uploaded image
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    result = cnn.predict(img_array)
    prediction = 'Cancer Present' if result[0][0] > 0.5 else 'No Cancer'
    
    st.write(f"Prediction: {prediction}")

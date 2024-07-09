from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import streamlit as st
import numpy as np
import os
from PIL import Image
import io

# Define paths to the datasets
training_set_path = r"C:\Users\parth sawant\Desktop\gemini compi\braintumor\trainingset"
test_set_path = r'C:\Users\parth sawant\Desktop\gemini compi\braintumor\testset'

# Preprocessing the Training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    training_set_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    test_set_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Building the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Create the Streamlit app
st.title("Brain Tumor Detection")
st.write("Upload an image to predict if there is a brain tumor.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load image data from file
    img = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    # Make prediction
    result = cnn.predict(img_array)
    prediction = 'Tumor Present' if result[0][0] > 0.5 else 'No Tumor'
    
    st.write(f"Prediction: {prediction}")

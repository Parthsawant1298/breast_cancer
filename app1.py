from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import base64
import io

app = Flask(__name__)

# Define paths to the datasets
training_set_path = "Breast Cancer Patients MRI/train"
test_set_path = "Breast Cancer Patients MRI/validation"

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
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

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

# Save the model

cnn.save("breastcancer.keras")

@app.route('/')
def home():
    return render_template('app.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('app.html', prediction_text='No file part')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('app.html', prediction_text='No selected file')
    
    if file:
        # Load image data from file
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert image to RGB if it is grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Preprocess the image
        img = img.resize((64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        
        # Make prediction
        result = cnn.predict(img_array)
        prediction = 'Breast cancer Present' if result[0][0] > 0.5 else 'No Breast cancer'
        
        # Display image in base64 format
        img_str = base64.b64encode(img_bytes).decode('utf-8')
        img_html = f'<img src="data:image/jpeg;base64,{img_str}" style="max-width: 400px;" />'
        
        return render_template('app.html', prediction_text=f'Prediction: {prediction}', image_html=img_html)

if __name__ == '__main__':
    app.run(debug=True, port=5001)

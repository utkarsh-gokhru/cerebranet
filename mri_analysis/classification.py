import cv2
import numpy as np
import tensorflow as tf
import os
from django.conf import settings

# Define your categories
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

import tempfile
import requests
import tensorflow as tf

def load_btc_model_from_firebase():
    # Firebase URL for your classification model
    firebase_url = 'https://firebasestorage.googleapis.com/v0/b/naac-fd101.appspot.com/o/models%2Fbtc.keras?alt=media&token=86c8acfe-0574-459e-b0cd-2afb937fe40d'
    
    # Create a temporary file to store the model
    temp_model_file = tempfile.NamedTemporaryFile(delete=False, suffix='.keras')

    # Download the model from Firebase
    response = requests.get(firebase_url)
    if response.status_code == 200:
        temp_model_file.write(response.content)
        temp_model_file.flush()
        temp_model_file.close()
        
        # Load the model from the temporary file
        model = tf.keras.models.load_model(temp_model_file.name)
        return model
    else:
        raise Exception(f"Failed to download model from Firebase: {response.status_code}")

# Load the BTC classification model
model = load_btc_model_from_firebase()


image_size = (256, 256)  # Image size used during training

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is not None:
        img = cv2.resize(img, image_size)  # Resize the image
        img = img / 255.0  # Normalize the image
        img = img.reshape(image_size[0], image_size[1], 1)  # Reshape to add the grayscale channel
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    else:
        raise ValueError("Error loading image. Please check the file path and format.")

def make_prediction(img_path):
    img_array = preprocess_image(img_path)
    prediction_probs = model.predict(img_array)  # Use 'model' here
    prediction_index = np.argmax(prediction_probs, axis=1)[0]
    prediction_label = categories[prediction_index]
    return prediction_label, prediction_probs[0]

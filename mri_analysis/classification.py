import cv2
import numpy as np
import os
from django.conf import settings

# Define your categories
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

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
    import tensorflow as tf  # <- Lazy load TensorFlow here
    model_path = os.path.join(settings.BASE_DIR, 'ml_models', 'btc.keras')
    model = tf.keras.models.load_model(model_path)  # Load the model only when needed

    img_array = preprocess_image(img_path)
    prediction_probs = model.predict(img_array)
    prediction_index = np.argmax(prediction_probs, axis=1)[0]
    prediction_label = categories[prediction_index]
    return prediction_label, prediction_probs[0]

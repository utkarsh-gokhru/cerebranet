import os
import cv2
import numpy as np
import tensorflow as tf
from django.core.files.storage import default_storage
from PIL import Image
from django.conf import settings
from firebase import upload_to_firebase
import tempfile, requests

# === Register custom Keras functions ===
@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# === Model load function with caching ===
MODEL_CACHE_PATH = '/tmp/unet_brain_mri_segmentation.keras'

def load_model_from_firebase():
    if os.path.exists(MODEL_CACHE_PATH):
        print("✅ Using cached model from /tmp")
        return tf.keras.models.load_model(
            MODEL_CACHE_PATH,
            custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient}
        )

    print("⏳ Downloading model from Firebase...")
    firebase_url = 'https://firebasestorage.googleapis.com/v0/b/naac-fd101.appspot.com/o/models%2Funet_brain_mri_segmentation.keras?alt=media&token=083944c4-a5a4-4ec3-8a22-1da373631617'

    response = requests.get(firebase_url)
    if response.status_code == 200:
        with open(MODEL_CACHE_PATH, 'wb') as f:
            f.write(response.content)

        return tf.keras.models.load_model(
            MODEL_CACHE_PATH,
            custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient}
        )
    else:
        raise Exception(f"Failed to download model from Firebase: {response.status_code}")

# === Load model once on startup ===
unet_model = load_model_from_firebase()

# === Image input size ===
image_size = (128, 128)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Error loading image. Please check the file path.")

    img = cv2.resize(img, image_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def calculate_tumor_size(mask, pixel_area_mm2=1.0):
    tumor_pixels = np.sum(mask > 0.5)
    tumor_size_mm2 = tumor_pixels * pixel_area_mm2
    return tumor_pixels, tumor_size_mm2

def run_segmentation(img_path):
    img = preprocess_image(img_path)
    pred_mask = unet_model.predict(img)[0]
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    return pred_mask, binary_mask

def visualize_segmentation(image, mask_pred):
    mask_pred_bin = (mask_pred > 0.5).astype(np.uint8)
    image_with_box = image.copy()
    contours, _ = cv2.findContours(mask_pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 0, 255), 2)

    mask_gray = (mask_pred.squeeze() * 255).astype(np.uint8)
    return image_with_box, mask_gray

def process_uploaded_image(uploaded_file):
    file_path = default_storage.save(f'uploads/{uploaded_file.name}', uploaded_file)
    absolute_path = os.path.join(settings.MEDIA_ROOT, file_path)

    pred_mask, binary_mask = run_segmentation(absolute_path)

    original_img = cv2.imread(absolute_path)
    original_img = cv2.resize(original_img, image_size)

    image_with_box, mask_gray = visualize_segmentation(original_img, pred_mask)

    filename = os.path.splitext(os.path.basename(uploaded_file.name))[0]
    box_name = f"{filename}_box.jpg"
    mask_name = f"{filename}_mask.jpg"
    original_name = f"{filename}_original.jpg"

    box_rel_path = os.path.join('with_bounding_box', box_name)
    mask_rel_path = os.path.join('tumor_mask', mask_name)
    original_rel_path = os.path.join('original_mri', original_name)

    box_temp_path = os.path.join(settings.MEDIA_ROOT, box_rel_path)
    mask_temp_path = os.path.join(settings.MEDIA_ROOT, mask_rel_path)
    original_temp_path = os.path.join(settings.MEDIA_ROOT, original_rel_path)

    os.makedirs(os.path.dirname(box_temp_path), exist_ok=True)
    os.makedirs(os.path.dirname(mask_temp_path), exist_ok=True)
    os.makedirs(os.path.dirname(original_temp_path), exist_ok=True)

    cv2.imwrite(box_temp_path, image_with_box)
    cv2.imwrite(mask_temp_path, mask_gray)
    cv2.imwrite(original_temp_path, original_img)

    original_firebase_url = upload_to_firebase(original_temp_path, f"mri/{os.path.basename(original_temp_path)}")
    mask_firebase_url = upload_to_firebase(mask_temp_path, f"tumor_mask/{os.path.basename(mask_temp_path)}")
    box_firebase_url = upload_to_firebase(box_temp_path, f"with_bounded_box/{os.path.basename(box_temp_path)}")

    os.remove(box_temp_path)
    os.remove(mask_temp_path)
    os.remove(original_temp_path)

    tumor_pixels, tumor_size_mm2 = calculate_tumor_size(pred_mask)

    return box_firebase_url, mask_firebase_url, original_firebase_url, tumor_pixels, tumor_size_mm2

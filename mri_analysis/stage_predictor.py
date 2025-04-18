import joblib
import numpy as np
import os
from django.conf import settings

# Load model and encoders
MODEL_DIR = os.path.join(settings.BASE_DIR, 'ml_models')  # adjust app name if needed

model = joblib.load(os.path.join(MODEL_DIR, 'xgb_tumor_stage_model.pkl'))
stage_encoder = joblib.load(os.path.join(MODEL_DIR, 'stage_label_encoder.pkl'))
tumor_type_categories = joblib.load(os.path.join(MODEL_DIR, 'tumor_type_categories.pkl'))

def predict_stage(input_data):
    """
    Accepts a dictionary of input values.
    Returns predicted stage as string (e.g., 'Stage 2')
    """
    tumor_type_encoded = tumor_type_categories.index(input_data['tumor_type'])

    features = np.array([[
        int(input_data['gender']),
        input_data['age'],
        input_data['tumor_size_cm'],
        tumor_type_encoded,
        input_data['crp_level'],
        input_data['ldh_level'],
        input_data['symptom_duration_months'],
        input_data['headache_frequency_per_week'],
        input_data['ki67_index_percent'],
        input_data['edema_volume_ml']
    ]])

    predicted_stage_num = model.predict(features)[0]
    predicted_stage = stage_encoder.inverse_transform([predicted_stage_num])[0]
    return predicted_stage

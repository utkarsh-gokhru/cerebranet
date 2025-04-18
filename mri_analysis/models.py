from django.db import models
from django.utils import timezone

class MRIImage(models.Model):
    image = models.ImageField(upload_to='mri_images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class AnalysisResult(models.Model):
    mri_image = models.ForeignKey(MRIImage, on_delete=models.CASCADE)
    tumor_detected = models.BooleanField()
    analyzed_at = models.DateTimeField(default=timezone.now)

    detailed_analysis = models.TextField(blank=True, null=True)
    
    # Store URLs instead of ImageField
    bounded_box_image_url = models.URLField(blank=True, null=True)
    segmented_result_url = models.URLField(blank=True, null=True)
    original_mri_image_url = models.URLField(blank=True, null=True)

    tumor_size_px = models.IntegerField(blank=True, null=True)
    tumor_size_mm2 = models.FloatField(blank=True, null=True)

    # Fields for tumor stage prediction
    gender = models.IntegerField(choices=[(0, 'Male'), (1, 'Female')], blank=True, null=True)
    age = models.IntegerField(blank=True, null=True)
    tumor_size_cm = models.FloatField(blank=True, null=True)
    crp_level = models.FloatField(blank=True, null=True)
    ldh_level = models.FloatField(blank=True, null=True)
    symptom_duration_months = models.FloatField(blank=True, null=True)
    headache_frequency_per_week = models.IntegerField(blank=True, null=True)
    ki67_index_percent = models.FloatField(blank=True, null=True)
    edema_volume_ml = models.FloatField(blank=True, null=True)

    stage = models.CharField(max_length=20, blank=True, null=True)

class Doctor(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    phone_number = models.CharField(max_length=15)
    password = models.CharField(max_length=255)
    doctor_reg_number = models.CharField(max_length=50)
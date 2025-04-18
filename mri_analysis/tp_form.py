from django import forms

TUMOR_TYPE_CHOICES = [
    ('glioma', 'Glioma'),
    ('meningioma', 'Meningioma'),
    ('pituitary', 'Pituitary'),
]

class TumorPredictionForm(forms.Form):
    gender = forms.ChoiceField(choices=[(0, 'Male'), (1, 'Female')])
    age = forms.IntegerField(min_value=1, max_value=120)
    tumor_size_cm = forms.FloatField()
    tumor_type = forms.ChoiceField(choices=TUMOR_TYPE_CHOICES)
    crp_level = forms.FloatField()
    ldh_level = forms.FloatField()
    symptom_duration_months = forms.FloatField()
    headache_frequency_per_week = forms.IntegerField()
    ki67_index_percent = forms.FloatField()
    edema_volume_ml = forms.FloatField()

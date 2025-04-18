import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import default_storage
from django.conf import settings
from PIL import Image
from .models import MRIImage, AnalysisResult
from .classification import make_prediction  # Import classification logic
from .segmentation import process_uploaded_image  # Updated function call
from .fuzzy_logic import assess_brain_score
from .bs_form import BrainScoreForm
from .stage_predictor import predict_stage
from .tp_form import TumorPredictionForm
from django.http import HttpResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Doctor
import hashlib, random
from twilio.rest import Client
from django.conf import settings
from django.contrib.auth.hashers import make_password, check_password

def landingPage(request):
    return render(request, 'index.html')

def auth(request):
    return render(request, 'signin.html')

# Signup logic
def signup(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        doctor_reg_number = request.POST.get('doctor_reg_number')

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('signup')

        if Doctor.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return redirect('signup')

        # Save doctor
        doctor = Doctor(
            name=name,
            email=email,
            phone_number=phone_number,
            password=make_password(password),  # Hashed password
            doctor_reg_number=doctor_reg_number
        )
        doctor.save()
        print('signup successful!')

        messages.success(request, "Account created successfully. Please log in.")
        return redirect('signin')
    else:
        return redirect('signup')

# Signin logic
def signin(request):
    print("[DEBUG] signin() view called")  
    if request.method == 'POST':
        print("[DEBUG] POST request received on /signin")

        unique_id = request.POST.get('unique_id')  # Either email or reg number
        password = request.POST.get('password')

        print(f"[DEBUG] unique_id: {unique_id}")
        print(f"[DEBUG] password: {password}")

        doctor = None
        try:
            doctor = Doctor.objects.get(email=unique_id)
            print("[DEBUG] Doctor found by email")
        except Doctor.DoesNotExist:
            print("[DEBUG] No doctor found with that email")
            try:
                doctor = Doctor.objects.get(doctor_reg_number=unique_id)
                print("[DEBUG] Doctor found by registration number")
            except Doctor.DoesNotExist:
                print("[DEBUG] No doctor found with that reg number")
                messages.error(request, "No account found with provided ID.")
                return redirect('auth')  # Make sure this redirect leads to the correct page (auth page)

        if check_password(password, doctor.password):
            print(f"[DEBUG] Password matched for Dr. {doctor.name}")
            request.session['doctor_id'] = doctor.id
            messages.success(request, f"Welcome back, Dr. {doctor.name}!")
            return redirect('brain_score')  # Redirect to brain score after successful login
        else:
            print("[DEBUG] Password mismatch")
            messages.error(request, "Invalid password.")
            return redirect('signin')  # Redirect back to signin if password is incorrect
    else:
        print("[DEBUG] GET request or non-POST request received on /signin")
        return render(request, 'signin.html')  # Render the signin page for non-POST requests


# Logout
def logout_view(request):
    request.session.flush()
    messages.success(request, "Logged out successfully.")
    return redirect('signin')

# Brain Score Assessment
def brain_score_view(request):
    result = None
    score = None

    if request.method == 'POST':
        form = BrainScoreForm(request.POST)
        if form.is_valid():
            data = list(map(int, form.cleaned_data.values()))
            score = assess_brain_score(data)
            if score >= 50:
                result = "Brain Score is Good 🙂"
            else:
                result = "Brain Score is Bad 🙁 (Recommendation: MRI Test)"
    else:
        form = BrainScoreForm()

    return render(request, 'brain_score.html', {'form': form, 'result': result, 'score': score})

def analyze_mri(request):
    if request.method == 'POST' and 'mri_image' in request.FILES:
        uploaded_file = request.FILES['mri_image']
        mri_image = MRIImage.objects.create(image=uploaded_file)

        # Step 1: Tumor Classification
        prediction_label, confidence_score = make_prediction(mri_image.image.path)
        print(f"Prediction Label: {prediction_label}, Confidence: {confidence_score}")

        tumor_detected = prediction_label.lower() != 'notumor'
        print(f"Tumor Detected: {tumor_detected}")

        # Step 2: Create initial analysis record
        analysis_result = AnalysisResult.objects.create(
            mri_image=mri_image,
            tumor_detected=tumor_detected,
            detailed_analysis=prediction_label
        )

        # Step 3: Run segmentation if tumor detected
        if tumor_detected:
            try:
                # Run segmentation and get visual results
                box_firebase_url, mask_firebase_url, original_firebase_url, tumor_pixels, tumor_size_mm2 = process_uploaded_image(mri_image.image)

                # Save Firebase URLs to the AnalysisResult
                analysis_result.bounded_box_image_url = box_firebase_url
                analysis_result.segmented_result_url = mask_firebase_url
                analysis_result.original_mri_image_url = original_firebase_url

                # Update tumor size
                analysis_result.tumor_size_px = tumor_pixels
                analysis_result.tumor_size_mm2 = tumor_size_mm2

            except Exception as e:
                print(f"Segmentation Error: {e}")

        # Step 4: Save and redirect
        analysis_result.save()

        return redirect('result_view', pk=analysis_result.pk)

    return render(request, 'upload_mri.html')

def result_view(request, pk):
    analysis_result = get_object_or_404(AnalysisResult, pk=pk)

    print(f"Analysis Result ID: {analysis_result.pk}, Tumor Detected: {analysis_result.tumor_detected}")

    return render(request, 'result_view.html', {'analysis_result': analysis_result})

def predict_tumor_stage(request):
    prediction = None
    analysis_result = None

    result_id = request.GET.get('result_id')  # This must be passed in the URL

    if result_id:
        # Fetch the corresponding analysis result using the provided ID
        analysis_result = get_object_or_404(AnalysisResult, pk=result_id)
        tumor_type = analysis_result.detailed_analysis
        tumor_size_mm2 = analysis_result.tumor_size_mm2

        # Convert mm² to cm² (1 cm² = 100 mm²)
        try:
            tumor_size_cm2 = round(float(tumor_size_mm2) / 100, 2)
        except (ValueError, TypeError):
            tumor_size_cm2 = ''
    else:
        tumor_type = ''
        tumor_size_cm2 = ''

    initial_data = {
        'tumor_type': tumor_type,
        'tumor_size_cm': tumor_size_cm2,
    }

    if request.method == 'POST':
        form = TumorPredictionForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            # Call your prediction function
            prediction = predict_stage(cd)

            if analysis_result:
                # Update analysis_result with form values
                analysis_result.gender = cd['gender']
                analysis_result.age = cd['age']
                analysis_result.tumor_size_cm = cd['tumor_size_cm']
                analysis_result.crp_level = cd['crp_level']
                analysis_result.ldh_level = cd['ldh_level']
                analysis_result.symptom_duration_months = cd['symptom_duration_months']
                analysis_result.headache_frequency_per_week = cd['headache_frequency_per_week']
                analysis_result.ki67_index_percent = cd['ki67_index_percent']
                analysis_result.edema_volume_ml = cd['edema_volume_ml']
                analysis_result.stage = prediction  # Save the predicted tumor stage
                analysis_result.save()

            return render(request, 'stage_predictor.html', {'form': form, 'prediction': prediction, 'result_id' : result_id})

    else:
        form = TumorPredictionForm(initial=initial_data)

    return render(request, 'stage_predictor.html', {'form': form, 'prediction': prediction, 'result_id' : result_id})

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import cm
from datetime import datetime
import os

import requests
from tempfile import NamedTemporaryFile

def generate_pdf(request):
    if request.method == 'POST':
        result_id = request.POST.get('result_id')
        if not result_id:
            return HttpResponse("Missing result ID", status=400)

        try:
            analysis_result = AnalysisResult.objects.get(pk=result_id)
        except AnalysisResult.DoesNotExist:
            return HttpResponse("Result not found", status=404)

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="tumor_report_{result_id}.pdf"'
        p = canvas.Canvas(response, pagesize=A4)

        width, height = A4
        margin = 50
        y = height - margin

        # ===== HEADER =====
        p.setFont("Helvetica-Bold", 22)
        p.setFillColor(colors.darkblue)
        p.drawCentredString(width / 2, y, "Tumor Analysis Report")
        y -= 30

        p.setFont("Helvetica", 10)
        timestamp = datetime.now().strftime("%d %B %Y, %I:%M %p")
        p.drawRightString(width - margin, y, f"Generated on: {timestamp}")
        y -= 20

        p.setStrokeColor(colors.darkblue)
        p.setLineWidth(1.2)
        p.line(margin, y, width - margin, y)
        y -= 30

        # ===== PATIENT DETAILS =====
        p.setFont("Helvetica-Bold", 14)
        p.setFillColor(colors.black)
        p.drawString(margin, y, "Patient & Tumor Details:")
        y -= 20
        p.setFont("Helvetica", 12)

        def draw_info(label, value, bold=False):
            nonlocal y
            if y < 150:
                p.showPage()
                y = height - margin
            p.setFont("Helvetica-Bold" if bold else "Helvetica", 12)
            p.drawString(margin + 10, y, f"{label}: {value}")
            y -= 20

        draw_info("Patient ID", analysis_result.id)
        draw_info("Gender", "Male" if analysis_result.gender == 0 else "Female")
        draw_info("Age", analysis_result.age)
        draw_info("CRP Level", analysis_result.crp_level)
        draw_info("LDH Level", analysis_result.ldh_level)
        draw_info("Symptom Duration (months)", analysis_result.symptom_duration_months)
        draw_info("Headache Frequency (per week)", analysis_result.headache_frequency_per_week)
        draw_info("Ki-67 Index (%)", analysis_result.ki67_index_percent)
        draw_info("Edema Volume (ml)", analysis_result.edema_volume_ml)
        draw_info("Tumor Type", analysis_result.detailed_analysis, bold=True)
        draw_info("Tumor Size (cm²)", analysis_result.tumor_size_cm, bold=True)
        draw_info("Predicted Tumor Stage", analysis_result.stage, bold=True)

        p.setStrokeColor(colors.grey)
        p.line(margin, y, width - margin, y)
        y -= 30

        # ===== IMAGES =====
        p.setFont("Helvetica-Bold", 14)
        p.drawString(margin, y, "Visual Analysis:")
        y -= 25

        def draw_image_from_path(image_path, label):
            nonlocal y
            try:
                if image_path and os.path.exists(image_path):
                    if y < 250:
                        p.showPage()
                        y = height - margin
                    p.setFont("Helvetica-Bold", 12)
                    p.drawString(margin, y, f"{label}:")
                    y -= 10
                    img_width = 10 * cm
                    img_height = 7.5 * cm
                    p.drawImage(image_path, margin, y - img_height, width=img_width, height=img_height)
                    y -= img_height + 30
            except Exception as e:
                print(f"Error displaying {label}: {e}")

        def download_temp_image(url):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    temp_file = NamedTemporaryFile(delete=False, suffix='.jpg')
                    temp_file.write(response.content)
                    temp_file.close()
                    return temp_file.name
            except Exception as e:
                print(f"Error downloading image: {e}")
            return None

        # MRI Image from local path
        if analysis_result.mri_image and analysis_result.mri_image.image:
            draw_image_from_path(analysis_result.mri_image.image.path, "MRI Image")

        # Segmented Result from Firebase URL
        if analysis_result.segmented_result_url:
            temp_seg_path = download_temp_image(analysis_result.segmented_result_url)
            if temp_seg_path:
                draw_image_from_path(temp_seg_path, "Segmented Result")

        # Bounded Box from Firebase URL
        if analysis_result.bounded_box_image_url:
            temp_box_path = download_temp_image(analysis_result.bounded_box_image_url)
            if temp_box_path:
                draw_image_from_path(temp_box_path, "Bounded Box Image")

        p.showPage()
        p.save()
        return response

    return HttpResponse("Invalid request method", status=405)

# Generated by Django 5.0.6 on 2024-09-25 13:13

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MRIImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='mri_images/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='AnalysisResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tumor_detected', models.BooleanField(default=False)),
                ('detailed_analysis', models.TextField(blank=True, null=True)),
                ('mri_image', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mri_analysis.mriimage')),
            ],
        ),
    ]

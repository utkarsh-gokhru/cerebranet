# Generated by Django 5.0.6 on 2024-10-21 05:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mri_analysis', '0002_analysisresult_analyzed_at_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='analysisresult',
            name='segmented_result',
            field=models.ImageField(blank=True, null=True, upload_to='segmentation_results/'),
        ),
    ]

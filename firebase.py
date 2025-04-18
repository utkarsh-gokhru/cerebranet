# firebase_init.py
import firebase_admin
from firebase_admin import credentials, storage
import os
from django.conf import settings

# Path to your downloaded Firebase service account key
cred_path = os.path.join(settings.BASE_DIR, './naac-fd101-firebase-adminsdk-szkmf-c408050452.json')

# Initialize Firebase app only once
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'naac-fd101.appspot.com'
    })

bucket = storage.bucket()

def upload_to_firebase(local_path, remote_path):
    """ Upload a file to Firebase Storage """
    try:
        # Create a blob object using the bucket
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        blob.make_public()  # Optionally make the file public
        return blob.public_url  # Return public URL of the uploaded file
    except Exception as e:
        print(f"Error uploading to Firebase: {e}")
        return None

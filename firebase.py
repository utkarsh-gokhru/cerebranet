import json
import firebase_admin
from firebase_admin import credentials, storage
import os
from django.conf import settings

# Fetch Firebase credentials from environment variable
firebase_service_account = os.getenv('FIREBASE_SERVICE_ACCOUNT')

# Parse the JSON string into a dictionary
cred_dict = json.loads(firebase_service_account)

# Initialize Firebase app only once
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_dict)
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

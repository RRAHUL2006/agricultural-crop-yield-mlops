#!/bin/sh

set -e

echo "Starting model download..."

# Initialize DVC without Git
dvc init --no-scm

# Add Google Drive remote
dvc remote add -f myremote gdrive://1T_8NdwTpFWIqz3aE_xMibUbwk1VgdexB

# Configure service account
dvc remote modify --local myremote gdrive_use_service_account true
dvc remote modify --local myremote gdrive_service_account_json_file_path /etc/secrets/service-account.json

# Download models
dvc pull models/best_model.pkl.dvc
dvc pull models/preprocessor.pkl.dvc

echo "Models downloaded successfully"

exec uvicorn app:app --host 0.0.0.0 --port 8000

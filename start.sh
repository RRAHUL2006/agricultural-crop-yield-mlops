#!/bin/sh

set -e

echo "Starting model download..."

dvc remote modify --local myremote gdrive_use_service_account true
dvc remote modify --local myremote gdrive_service_account_json_file_path /etc/secrets/service-account.json

dvc pull models/best_model.pkl.dvc
dvc pull models/preprocessor.pkl.dvc

echo "Models downloaded successfully"

exec uvicorn app:app --host 0.0.0.0 --port 8000

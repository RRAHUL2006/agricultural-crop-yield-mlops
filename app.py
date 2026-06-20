from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os


# Create FastAPI application
app = FastAPI(
    title="Agricultural Crop Yield Prediction API",
    description="Predict crop yield using weather, soil, and farming data",
    version="1.0"
)

model = None
preprocessor = None


def load_models():
    global model, preprocessor

    print("Configuring DVC authentication...")

    os.system(
        "dvc remote modify --local myremote "
        "gdrive_use_service_account true"
    )

    service_account_path = (
        "/etc/secrets/service-account.json"
        if os.path.exists("/etc/secrets/service-account.json")
        else "service-account.json"
    )

    os.system(
        f"dvc remote modify --local myremote "
        f"gdrive_service_account_json_file_path {service_account_path}"
    )

    print("Downloading models from DVC...")

    if os.system("dvc pull models/best_model.pkl.dvc") != 0:
        raise Exception("Failed to download best_model.pkl")

    if os.system("dvc pull models/preprocessor.pkl.dvc") != 0:
        raise Exception("Failed to download preprocessor.pkl")

    print("Loading models...")

    model = joblib.load("models/best_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")

    print("Models loaded successfully")
    
load_models()
# Input schema
class CropInput(BaseModel):
    crop: str
    year: int
    season: str
    state: str
    area: float
    fertilizer: float
    pesticide: float
    avg_temp_c: float
    total_rainfall_mm: float
    avg_humidity_percent: float
    N: float
    P: float
    K: float
    pH: float


# Home route
@app.get("/")
def home():
    return {
        "message": "Agricultural Crop Yield Prediction API is running"
    }


# Prediction route
@app.post("/predict")
def predict(data: CropInput):
    if model is None or preprocessor is None:
        return {"error": "Model files not available"}
    input_data = pd.DataFrame(
        [data.model_dump()]
    )

    transformed_data = preprocessor.transform(
        input_data
    )

    prediction = model.predict(
        transformed_data
    )

    return {
        "predicted_yield": round(
            float(prediction[0]),
            2
        )
    }
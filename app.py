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


import gdown

model = None
preprocessor = None

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

MODEL_ID = "1iCACPhCp6FyCYxygDFq9dPCU3AUvYQ6C"
PREPROCESSOR_ID = "1tK3xmynzHHxBkfYYgFvnN-Q-H_XRuywD"


def download_models():
    if not os.path.exists(MODEL_PATH):
        print("Downloading best_model.pkl...")
        gdown.download(
            f"https://drive.google.com/uc?id={MODEL_ID}",
            MODEL_PATH,
            quiet=False
        )

    if not os.path.exists(PREPROCESSOR_PATH):
        print("Downloading preprocessor.pkl...")
        gdown.download(
            f"https://drive.google.com/uc?id={PREPROCESSOR_ID}",
            PREPROCESSOR_PATH,
            quiet=False
        )


download_models()

print("Loading models...")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Models loaded successfully")



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
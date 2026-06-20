from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# Create FastAPI application
app = FastAPI(
    title="Agricultural Crop Yield Prediction API",
    description="Predict crop yield using weather, soil, and farming data",
    version="1.0"
)


# Load trained artifacts
model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


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
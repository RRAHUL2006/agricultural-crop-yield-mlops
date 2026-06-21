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

if os.path.exists("models/best_model.pkl") and os.path.exists("models/preprocessor.pkl"):
    print("Loading models...")

    model = joblib.load("models/best_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")

    print("Models loaded successfully")
else:
    print("Model files not found")





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
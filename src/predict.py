import joblib
import pandas as pd

# Load artifacts
model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

sample = pd.DataFrame([
    {
        "crop": "Rice",
        "year": 2021,
        "season": "Kharif",
        "state": "Tamil Nadu",
        "area": 100,
        "fertilizer": 5000,
        "pesticide": 100,
        "avg_temp_c": 28,
        "total_rainfall_mm": 1200,
        "avg_humidity_percent": 75,
        "N": 80,
        "P": 40,
        "K": 30,
        "pH": 6.8
    }
])

X = preprocessor.transform(sample)
prediction = model.predict(X)

print(f"Predicted Yield: {prediction[0]:.2f}")
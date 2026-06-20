import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor():

    categorical_features = [
        "crop",
        "season",
        "state"
    ]

    numerical_features = [
        "year",
        "area",
        "fertilizer",
        "pesticide",
        "avg_temp_c",
        "total_rainfall_mm",
        "avg_humidity_percent",
        "N",
        "P",
        "K",
        "pH"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "num",
                StandardScaler(),
                numerical_features
            )
        ]
    )

    return preprocessor


def main():

    print("=" * 50)
    print("PREPROCESSING PIPELINE CREATION")
    print("=" * 50)

    df = pd.read_csv(
        "data/processed/final_dataset.csv"
    )

    X = df.drop(
        columns=["yield"]
    )

    y = df["yield"]

    preprocessor = create_preprocessor()

    X_transformed = preprocessor.fit_transform(X)

    print("\nOriginal shape:")
    print(X.shape)

    print("\nTransformed shape:")
    print(X_transformed.shape)

    print("\nSaving preprocessing pipeline...")

    joblib.dump(
        preprocessor,
        "models/preprocessor.pkl"
    )

    print("\nSUCCESS: Preprocessor saved to models/preprocessor.pkl")


if __name__ == "__main__":
    main()
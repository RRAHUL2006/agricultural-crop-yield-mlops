import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor


def evaluate_model(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }


def main():

    print("=" * 50)
    print("MODEL TRAINING STARTED")
    print("=" * 50)

    # Load data
    df = pd.read_csv(
        "data/processed/final_dataset.csv"
    )

    X = df.drop("yield", axis=1)
    y = df["yield"]

    # Load preprocessing pipeline
    preprocessor = joblib.load(
        "models/preprocessor.pkl"
    )

    X = preprocessor.transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            random_state=42
        )
    }

    best_score = -999
    best_model = None

    for name, model in models.items():

        print(f"\nTraining {name}...")

        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            metrics = evaluate_model(
                y_test,
                predictions
            )

            print(metrics)

            mlflow.log_params(
                model.get_params()
            )

            mlflow.log_metrics(
                metrics
            )

            if name == "XGBoost":
                mlflow.xgboost.log_model(
                    model,
                    name=name
    )
            else:
                mlflow.sklearn.log_model(
        model,
        name=name
    )

            if metrics["R2"] > best_score:
                best_score = metrics["R2"]
                best_model = model


    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best R2 Score: {best_score}")
    print("Best model saved to models/best_model.pkl")


if __name__ == "__main__":
    main()
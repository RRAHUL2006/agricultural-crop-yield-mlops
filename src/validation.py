import pandas as pd


def validate_data(df):
    errors = []

    # Check missing values
    if df.isnull().sum().sum() > 0:
        errors.append("Missing values found")

    # Check duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        errors.append(f"{duplicates} duplicate rows found")

    # Check area
    if (df["area"] <= 0).any():
        errors.append("Invalid area values found")

    # Check fertilizer
    if (df["fertilizer"] < 0).any():
        errors.append("Negative fertilizer values found")

    # Check pesticide
    if (df["pesticide"] < 0).any():
        errors.append("Negative pesticide values found")

    # Check soil pH
    if ((df["pH"] < 0) | (df["pH"] > 14)).any():
        errors.append("Invalid pH values found")

    # Check temperature
    if ((df["avg_temp_c"] < -10) | (df["avg_temp_c"] > 60)).any():
        errors.append("Abnormal temperature values found")

    return errors


def main():

    df = pd.read_csv(
        "data/processed/final_dataset.csv"
    )

    errors = validate_data(df)

    print("="*50)
    print("DATA VALIDATION REPORT")
    print("="*50)

    if len(errors) == 0:
        print("SUCCESS: Dataset passed all validation checks")
    else:
        print("FAILED: Dataset has problems")
        for error in errors:
            print("-", error)


if __name__ == "__main__":
    main()
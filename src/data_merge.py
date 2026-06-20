import pandas as pd

# Load datasets
crop = pd.read_csv("data/raw/crop_yield.csv")
soil = pd.read_csv("data/raw/state_soil_data.csv")
weather = pd.read_csv("data/raw/state_weather_data_1997_2020.csv")


# Merge crop + weather using state and year
data = crop.merge(
    weather,
    on=["state", "year"],
    how="left"
)


# Merge soil using state
data = data.merge(
    soil,
    on="state",
    how="left"
)


# Remove data leakage
data.drop(
    columns=["production"],
    inplace=True
)


# Check merged data
print("Final shape:", data.shape)
print("\nColumns:")
print(data.columns)

print("\nMissing values:")
print(data.isnull().sum())


# Save processed dataset
data.to_csv(
    "data/processed/final_dataset.csv",
    index=False
)

print("\nDataset saved successfully!")
import pandas as pd

# Load datasets
crop = pd.read_csv("data/raw/crop_yield.csv")
soil = pd.read_csv("data/raw/state_soil_data.csv")
weather = pd.read_csv("data/raw/state_weather_data_1997_2020.csv")

# Display information
print("\n===== CROP DATA =====")
print(crop.head())
print("\nColumns:")
print(crop.columns)

print("\n===== SOIL DATA =====")
print(soil.head())
print("\nColumns:")
print(soil.columns)

print("\n===== WEATHER DATA =====")
print(weather.head())
print("\nColumns:")
print(weather.columns)
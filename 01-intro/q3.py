from pathlib import Path

import pandas as pd

df = pd.read_csv(Path(__file__).parent / "data/car_fuel_efficiency_2026.csv")

print(df["fuel_type"].value_counts())
print("Q3. Fuel types:", df["fuel_type"].nunique())

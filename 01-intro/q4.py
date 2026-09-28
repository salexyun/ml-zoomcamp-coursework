from pathlib import Path

import pandas as pd

df = pd.read_csv(Path(__file__).parent / "data/car_fuel_efficiency_2026.csv")

missing = df.isnull().sum()
print(missing[missing > 0])
print("Q4. Columns with missing values:", (missing > 0).sum())

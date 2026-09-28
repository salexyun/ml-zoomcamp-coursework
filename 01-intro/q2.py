from pathlib import Path

import pandas as pd

df = pd.read_csv(Path(__file__).parent / "data/car_fuel_efficiency_2026.csv")

print("Q2. Records count:", len(df))

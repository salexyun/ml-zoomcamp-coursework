from pathlib import Path

import pandas as pd

df = pd.read_csv(Path(__file__).parent / "data/car_fuel_efficiency_2026.csv")

asia = df[df["origin"] == "Asia"]
print("Q5. Max fuel efficiency (Asia):", asia["fuel_efficiency_mpg"].max())

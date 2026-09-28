from pathlib import Path

import pandas as pd

df = pd.read_csv(Path(__file__).parent / "data/car_fuel_efficiency_2026.csv")

median_before = df["horsepower"].median()
most_frequent = df["horsepower"].mode()[0]
median_after = df["horsepower"].fillna(most_frequent).median()

print("Median before:", median_before)
print("Most frequent:", most_frequent)
print("Median after:", median_after)

if median_after > median_before:
    print("Q6. Yes, it increased")
elif median_after < median_before:
    print("Q6. Yes, it decreased")
else:
    print("Q6. No")

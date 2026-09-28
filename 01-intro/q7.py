from pathlib import Path

import numpy as np
import pandas as pd

df = pd.read_csv(Path(__file__).parent / "data/car_fuel_efficiency_2026.csv")

X = df[df["origin"] == "Asia"][["vehicle_weight", "model_year"]].head(7).values
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y

print("w:", w)
print("Q7. Sum of weights:", w.sum())

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["engine_displacement", "horsepower", "vehicle_weight", "model_year"]
TARGET = "fuel_efficiency_mpg"


def load_data():
    df = pd.read_csv(Path(__file__).parent / "data/car_fuel_efficiency_2026.csv")
    return df[FEATURES + [TARGET]]


def split(df, seed=42):
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]
    return df_train, df_val, df_test


def prepare_X(df, fill_value=0):
    return df[FEATURES].fillna(fill_value).values


def train_linear_regression(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T @ X
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv @ X.T @ y
    return w_full[0], w_full[1:]


def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))

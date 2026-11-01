from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

TARGET = "fuel_efficiency_mpg"


def load_data():
    df = pd.read_csv(Path(__file__).parent / "data/car_fuel_efficiency_2026.csv")
    return df.fillna(0)


def split(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    return df_full_train, df_train, df_val, df_test


def prepare():
    _, df_train, df_val, _ = split(load_data())
    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(df_train.drop(columns=TARGET).to_dict(orient="records"))
    X_val = dv.transform(df_val.drop(columns=TARGET).to_dict(orient="records"))
    return dv, X_train, X_val, df_train[TARGET].values, df_val[TARGET].values


def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))

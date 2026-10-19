from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

CATEGORICAL = ["lead_source", "industry", "employment_status", "location"]
NUMERICAL = ["annual_income", "number_of_courses_viewed", "interaction_count", "lead_score"]
TARGET = "converted"
FEATURES = CATEGORICAL + NUMERICAL


def load_data():
    df = pd.read_csv(Path(__file__).parent / "data/course_lead_scoring_2026.csv")
    df[CATEGORICAL] = df[CATEGORICAL].fillna("NA")
    df[NUMERICAL] = df[NUMERICAL].fillna(0.0)
    return df


def split(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    return df_full_train, df_train, df_val, df_test


def train(df, y, C=1.0):
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(df[FEATURES].to_dict(orient="records"))
    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000)
    model.fit(X, y)
    return dv, model


def predict(df, dv, model):
    X = dv.transform(df[FEATURES].to_dict(orient="records"))
    return model.predict_proba(X)[:, 1]


def precision_recall(y_true, y_pred, thresholds):
    rows = []
    for t in thresholds:
        predicted = y_pred >= t
        tp = (predicted & (y_true == 1)).sum()
        fp = (predicted & (y_true == 0)).sum()
        fn = (~predicted & (y_true == 1)).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append((t, precision, recall))
    return pd.DataFrame(rows, columns=["threshold", "precision", "recall"])


THRESHOLDS = np.round(np.arange(0.0, 1.01, 0.01), 2)

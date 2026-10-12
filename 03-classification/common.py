from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CATEGORICAL = ["lead_source", "industry", "employment_status", "location"]
NUMERICAL = ["annual_income", "number_of_courses_viewed", "interaction_count", "lead_score"]
TARGET = "converted"


def load_data():
    df = pd.read_csv(Path(__file__).parent / "data/course_lead_scoring_2026.csv")
    df[CATEGORICAL] = df[CATEGORICAL].fillna("NA")
    df[NUMERICAL] = df[NUMERICAL].fillna(0.0)
    return df


def split(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

    y_train = df_train.pop(TARGET).values
    y_val = df_val.pop(TARGET).values
    y_test = df_test.pop(TARGET).values
    return df_train, df_val, df_test, y_train, y_val, y_test


def train_and_score(df_train, y_train, df_val, y_val, features, C=1.0):
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train[features].to_dict(orient="records"))
    X_val = dv.transform(df_val[features].to_dict(orient="records"))

    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return accuracy_score(y_val, model.predict(X_val))

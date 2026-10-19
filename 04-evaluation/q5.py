import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from common import TARGET, load_data, predict, split, train


def cv_scores(df_full_train, C=1.0):
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
        dv, model = train(df_train, df_train[TARGET].values, C=C)
        scores.append(roc_auc_score(df_val[TARGET], predict(df_val, dv, model)))
    return np.array(scores)


if __name__ == "__main__":
    df_full_train, _, _, _ = split(load_data())
    scores = cv_scores(df_full_train)
    print("Fold AUCs:", scores.round(3))
    print("Q5. Std of AUC:", round(scores.std(), 3))

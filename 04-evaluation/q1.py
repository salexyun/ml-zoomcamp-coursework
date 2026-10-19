from sklearn.metrics import roc_auc_score

from common import NUMERICAL, TARGET, load_data, split

df = load_data()
_, df_train, _, _ = split(df)

aucs = {}
for col in NUMERICAL:
    auc = roc_auc_score(df_train[TARGET], df_train[col])
    if auc < 0.5:
        auc = roc_auc_score(df_train[TARGET], -df_train[col])
    aucs[col] = auc
    print(f"{col}: {auc:.4f}")

print("Q1. Highest AUC:", max(aucs, key=aucs.get))

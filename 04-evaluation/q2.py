from sklearn.metrics import roc_auc_score

from common import TARGET, load_data, predict, split, train

df = load_data()
_, df_train, df_val, _ = split(df)

dv, model = train(df_train, df_train[TARGET].values)
y_pred = predict(df_val, dv, model)

print("Q2. Validation AUC:", round(roc_auc_score(df_val[TARGET], y_pred), 3))

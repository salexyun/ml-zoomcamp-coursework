from common import TARGET, THRESHOLDS, load_data, precision_recall, predict, split, train

df = load_data()
_, df_train, df_val, _ = split(df)

dv, model = train(df_train, df_train[TARGET].values)
y_pred = predict(df_val, dv, model)

scores = precision_recall(df_val[TARGET].values, y_pred, THRESHOLDS)
p, r = scores.precision, scores.recall
scores["f1"] = (2 * p * r / (p + r)).fillna(0.0)

print(scores.loc[scores.f1.nlargest(3).index])
print("Q4. Max F1 threshold:", scores.threshold[scores.f1.idxmax()])

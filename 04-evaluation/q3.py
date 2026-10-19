from common import TARGET, THRESHOLDS, load_data, precision_recall, predict, split, train

df = load_data()
_, df_train, df_val, _ = split(df)

dv, model = train(df_train, df_train[TARGET].values)
y_pred = predict(df_val, dv, model)

scores = precision_recall(df_val[TARGET].values, y_pred, THRESHOLDS)
scores = scores[(scores.precision > 0) | (scores.recall > 0)]
diff = (scores.precision - scores.recall).abs()

print(scores.loc[diff.nsmallest(3).index])
print("Q3. Intersection threshold:", scores.threshold[diff.idxmin()])

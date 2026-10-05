from common import TARGET, load_data, prepare_X, rmse, split, train_linear_regression

df = load_data()
df_train, df_val, _ = split(df, seed=42)
X_train = prepare_X(df_train)
X_val = prepare_X(df_val)
y_train = df_train[TARGET].values
y_val = df_val[TARGET].values

scores = {}
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w0, w = train_linear_regression(X_train, y_train, r=r)
    scores[r] = round(rmse(y_val, w0 + X_val @ w), 4)
    print(f"r={r}: {scores[r]}")

# min() keeps the first (smallest) r on ties
print("Q4. Best r:", min(scores, key=scores.get))

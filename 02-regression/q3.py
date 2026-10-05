from common import TARGET, load_data, prepare_X, rmse, split, train_linear_regression

df = load_data()
df_train, df_val, _ = split(df, seed=42)
y_train = df_train[TARGET].values
y_val = df_val[TARGET].values

fill_values = {
    "With 0": 0,
    "With mean": df_train["horsepower"].mean(),
}

scores = {}
for name, fill_value in fill_values.items():
    w0, w = train_linear_regression(prepare_X(df_train, fill_value), y_train)
    y_pred = w0 + prepare_X(df_val, fill_value) @ w
    scores[name] = round(rmse(y_val, y_pred), 3)
    print(f"{name}: {scores[name]}")

if len(set(scores.values())) == 1:
    print("Q3. Better option: Both are equally good")
else:
    print("Q3. Better option:", min(scores, key=scores.get))

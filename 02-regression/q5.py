import numpy as np

from common import TARGET, load_data, prepare_X, rmse, split, train_linear_regression

df = load_data()

scores = []
for seed in range(10):
    df_train, df_val, _ = split(df, seed=seed)
    w0, w = train_linear_regression(prepare_X(df_train), df_train[TARGET].values)
    score = rmse(df_val[TARGET].values, w0 + prepare_X(df_val) @ w)
    scores.append(score)
    print(f"seed={seed}: {score:.4f}")

print("Q5. Std of scores:", round(np.std(scores), 3))

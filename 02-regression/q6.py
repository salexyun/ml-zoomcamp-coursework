import pandas as pd

from common import TARGET, load_data, prepare_X, rmse, split, train_linear_regression

df = load_data()
df_train, df_val, df_test = split(df, seed=9)
df_full_train = pd.concat([df_train, df_val])

w0, w = train_linear_regression(prepare_X(df_full_train), df_full_train[TARGET].values, r=0.001)
score = rmse(df_test[TARGET].values, w0 + prepare_X(df_test) @ w)

print("Q6. Test RMSE:", round(score, 3))

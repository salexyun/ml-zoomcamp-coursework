from common import CATEGORICAL, NUMERICAL, load_data, split, train_and_score

df = load_data()
df_train, df_val, _, y_train, y_val, _ = split(df)

accuracy = train_and_score(df_train, y_train, df_val, y_val, CATEGORICAL + NUMERICAL)

print("Q4. Validation accuracy:", round(accuracy, 2))

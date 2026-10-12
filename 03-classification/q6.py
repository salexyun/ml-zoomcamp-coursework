from common import CATEGORICAL, NUMERICAL, load_data, split, train_and_score

df = load_data()
df_train, df_val, _, y_train, y_val, _ = split(df)

scores = {}
for C in [0.000001, 0.00001, 0.0001, 0.001]:
    scores[C] = round(train_and_score(df_train, y_train, df_val, y_val, CATEGORICAL + NUMERICAL, C=C), 3)
    print(f"C={C}: {scores[C]}")

# max() keeps the first (smallest) C on ties
print("Q6. Best C:", max(scores, key=scores.get))

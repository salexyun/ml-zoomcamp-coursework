from sklearn.metrics import mutual_info_score

from common import CATEGORICAL, load_data, split

df = load_data()
df_train, _, _, y_train, _, _ = split(df)

scores = {}
for col in CATEGORICAL:
    scores[col] = round(mutual_info_score(y_train, df_train[col]), 2)
    print(f"{col}: {scores[col]}")

print("Q3. Biggest mutual information:", max(scores, key=scores.get))

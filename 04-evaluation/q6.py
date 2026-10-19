from common import load_data, split
from q5 import cv_scores

df_full_train, _, _, _ = split(load_data())

results = {}
for C in [0.000001, 0.001, 1]:
    scores = cv_scores(df_full_train, C=C)
    results[C] = (round(scores.mean(), 3), round(scores.std(), 3))
    print(f"C={C}: mean={results[C][0]} std={results[C][1]}")

# highest mean, then lowest std, then smallest C
best = min(results, key=lambda C: (-results[C][0], results[C][1], C))
print("Q6. Best C:", best)

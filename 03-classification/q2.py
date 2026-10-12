from common import NUMERICAL, load_data

df = load_data()

corr = df[NUMERICAL].corr()
print(corr.round(4))

pairs = [
    ("interaction_count", "lead_score"),
    ("number_of_courses_viewed", "lead_score"),
    ("number_of_courses_viewed", "interaction_count"),
    ("annual_income", "interaction_count"),
]
scores = {pair: corr.loc[pair] for pair in pairs}
for (a, b), score in scores.items():
    print(f"{a} / {b}: {score:.4f}")

# biggest by absolute value
best = max(scores, key=lambda pair: abs(scores[pair]))
print("Q2. Most correlated pair:", " and ".join(best))

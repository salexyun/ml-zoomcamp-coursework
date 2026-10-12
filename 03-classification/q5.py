from common import CATEGORICAL, NUMERICAL, load_data, split, train_and_score

df = load_data()
df_train, df_val, _, y_train, y_val, _ = split(df)

features = CATEGORICAL + NUMERICAL
base = train_and_score(df_train, y_train, df_val, y_val, features)
print(f"All features: {base:.4f}")

diffs = {}
for feature in features:
    subset = [f for f in features if f != feature]
    accuracy = train_and_score(df_train, y_train, df_val, y_val, subset)
    diffs[feature] = base - accuracy
    print(f"without {feature}: {accuracy:.4f} (diff {diffs[feature]:+.4f})")

candidates = ["lead_source", "number_of_courses_viewed", "interaction_count"]
print("Q5. Smallest difference:", min(candidates, key=diffs.get))

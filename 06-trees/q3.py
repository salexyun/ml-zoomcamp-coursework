from sklearn.ensemble import RandomForestRegressor

from common import prepare, rmse

_, X_train, X_val, y_train, y_val = prepare()

scores = {}
for n in [10, 50, 100, 150]:
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1).fit(X_train, y_train)
    scores[n] = round(rmse(y_val, rf.predict(X_val)), 3)
    print(f"n_estimators={n}: {scores[n]}")

print("Q3. Best n_estimators:", min(scores, key=scores.get))

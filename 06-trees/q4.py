import numpy as np
from sklearn.ensemble import RandomForestRegressor

from common import prepare, rmse

_, X_train, X_val, y_train, y_val = prepare()

means = {}
for d in [10, 15, 20, 25]:
    scores = [
        rmse(y_val, RandomForestRegressor(n_estimators=n, max_depth=d, random_state=1, n_jobs=-1)
             .fit(X_train, y_train).predict(X_val))
        for n in [10, 50, 100, 150]
    ]
    means[d] = np.mean(scores)
    print(f"max_depth={d}: mean RMSE {means[d]:.4f}")

print("Q4. Best max_depth:", min(means, key=means.get))

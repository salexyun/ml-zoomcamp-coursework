from sklearn.ensemble import RandomForestRegressor

from common import prepare, rmse

_, X_train, X_val, y_train, y_val = prepare()

rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1).fit(X_train, y_train)
print("Q2. Validation RMSE:", round(rmse(y_val, rf.predict(X_val)), 4))

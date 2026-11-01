import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from common import prepare

dv, X_train, _, y_train, _ = prepare()

rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1).fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=dv.get_feature_names_out()).sort_values(ascending=False)
print(importances.round(4))

candidates = ["vehicle_weight", "horsepower", "acceleration", "engine_displacement"]
print("Q5. Most important feature:", importances[candidates].idxmax())

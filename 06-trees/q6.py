import xgboost as xgb

from common import prepare, rmse

dv, X_train, X_val, y_train, y_val = prepare()
features = list(dv.get_feature_names_out())

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
watchlist = [(dtrain, "train"), (dval, "val")]

scores = {}
for eta in [0.3, 0.1]:
    xgb_params = {
        "eta": eta,
        "max_depth": 6,
        "min_child_weight": 1,
        "objective": "reg:squarederror",
        "nthread": 8,
        "seed": 1,
        "verbosity": 1,
    }
    model = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist, verbose_eval=False)
    scores[eta] = round(rmse(y_val, model.predict(dval)), 4)
    print(f"eta={eta}: {scores[eta]}")

best = min(scores.values())
winners = [eta for eta, s in scores.items() if s == best]
print("Q6. Best eta:", winners[0] if len(winners) == 1 else "Both give equal value")

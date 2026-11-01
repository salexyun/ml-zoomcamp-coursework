from sklearn.tree import DecisionTreeRegressor, export_text

from common import prepare

dv, X_train, _, y_train, _ = prepare()
names = dv.get_feature_names_out()

dt = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)
print(export_text(dt, feature_names=list(names)))
print("Q1. Splitting feature:", names[dt.tree_.feature[0]].split("=")[0])

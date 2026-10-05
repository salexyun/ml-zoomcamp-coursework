from common import load_data

df = load_data()

missing = df.isnull().sum()
print(missing)
print("Q1. Column with missing values:", missing[missing > 0].index[0])

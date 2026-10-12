from common import load_data

df = load_data()

print(df["industry"].value_counts())
print("Q1. Mode of industry:", df["industry"].mode()[0])

import pandas as pd

df = pd.read_csv("tests/data/pm10.csv", index_col=0)


df_binary = df.copy()
# make this an indicator for binary classification
df_binary["predictand"] = df["PM10"] > 20

df = df[["x", "y", "time", "PM10"]]
df_binary = df_binary[["x", "y", "time", "predictand"]]

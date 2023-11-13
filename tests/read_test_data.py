import pandas as pd

df = pd.read_csv("tests/data/pm10.csv", index_col=0)

df["presence"] = df["PM10"] > 20

df = df[["x", "y", "time", "presence"]]

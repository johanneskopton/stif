import os
from pathlib import Path

import pyreadr


data_path = "https://github.com/edzer/sp/raw/main/data/meuse.rda"

Path("tests/data").mkdir(parents=True, exist_ok=True)
pyreadr.download_file(data_path, "tests/data/meuse.rds")

result = pyreadr.read_r("tests/data/meuse.rds")

result["meuse"].to_csv("tests/data/meuse.csv")

os.remove("tests/data/meuse.rds")

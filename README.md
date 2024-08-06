# FIST ✊

"Forecasting and Interpolation in Space and Time"

Package for prediction spatio-temporally distributed variables using space-time regression kriging using numpy and numba.

## Installation

Via pip:

```sh
pip install .
```

## Development

Installing via pip for development:

```sh
pip install -e .[dev]
```

Testing:

```sh
pytest --cov=fist --junitxml=docs/source/_static/reports/junit/junit.xml --html=docs/source/_static/reports/junit/report.html
```

Reports and badges:
```sh
coverage html -d docs/source/_static/reports/htmlcov
coverage xml -o docs/source/_static/reports/coverage.xml
flake8 --statistics --tee --output-file docs/source/_static/reports/flake8stats.txt
genbadge tests -i docs/source/_static/reports/junit/junit.xml -o docs/source/_static/tests-badge.svg
genbadge coverage -i docs/source/_static/reports/coverage.xml -o docs/source/_static/coverage-badge.svg
genbadge flake8 -i docs/source/_static/reports/flake8stats.txt -o docs/source/_static/flake8-badge.svg
```

Build sphinx documentation:
```sh
sphinx-build -M html docs/source/ docs/build/
```

## Acknowledgements
Heavily inspired by [^1] and the space-time variogram implementation in the R package [gstat](http://r-spatial.github.io/gstat/) [^2]. Thanks a lot to Sytze de Bruin from Wageningen University for a lot of help with the geostatistical details.

[^1]: G. B. M. Heuvelink, E. Pebesma, and B. Gräler, “Space-Time Geostatistics,” in Encyclopedia of GIS, S. Shekhar, H. Xiong, and X. Zhou, Eds., Cham: Springer International Publishing, 2017, pp. 1919–1926. doi: 10.1007/978-3-319-17885-1_1647.

[^2]: B. Graeler, E. Pebesma, and G. Heuvelink, “Spatio-Temporal Interpolation using gstat,” The R Journal, vol. 8, pp. 204–218, Jan. 2016, doi: 10.32614/RJ-2016-014.

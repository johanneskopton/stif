# FIST ✊

"Forecasting and Interpolation in Space and Time"

Package for predicting spatio-temporally distributed variables via space-time regression Kriging, using numpy and numba.

Using this package, you can do _space-time regression Kriging_. What is this method about? Imagine you have measurements on different days and distributed over a geographical region (basically a table with latitude, longitude, time and the measured variable of interest). And now you want to estimate the value of this variable at locations and dates, at which you didn't measure. This can be the future (forecasting) or just some day in between, where no measurement was taken (interpolation). You may have some other variables (external covariates), that can explain some of the variation of you variable of interest via some form of regression. In this case, it makes sense to do the Kriging only on the residuals of this regression (i.e. *regression Kriging*). All of this, you can do using this package (and hopefully as simple as possible).

## Why this package?
To my knowledge, there is no other Python package for space-time variogram Kriging on unstructured spatio-temporal data. While the R package [gstat](http://r-spatial.github.io/gstat/) [^1] can do space-time Kriging, there can be problems with large datasets. Furthermore, the R package obviously doesn't play well with the Python ecosystem, e.g. using Keras/tensorflow for the regression part of regression Kriging.

There is the excellent [scikit-gstat](https://github.com/mmaelicke/scikit-gstat) [^2], but as for now it can not create variograms for unstructured spatio-temporal data and I didn't have the time yet to dive into it and add this feature. So perhaps for now, you may find this package useful.

You can also do something like space-time regression Kriging using one of the more mature Gaussian process libraries out there (since Kriging is actually just Gaussian process regression), but they won't provide you with nice variograms and all the other geostats stuff.

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
flake8 --format=html --htmldir=docs/source/_static/reports/flake8
genbadge tests -i docs/source/_static/reports/junit/junit.xml -o docs/source/_static/tests-badge.svg
genbadge coverage -i docs/source/_static/reports/coverage.xml -o docs/source/_static/coverage-badge.svg
genbadge flake8 -i docs/source/_static/reports/flake8stats.txt -o docs/source/_static/flake8-badge.svg
```

Check dependencies:
```sh
tox
```

Build sphinx documentation:
```sh
sphinx-build -M html docs/source/ docs/build/
```

## Acknowledgements
Heavily inspired by [^1] and the space-time variogram implementation in the R package [gstat](http://r-spatial.github.io/gstat/) [^2]. Thanks a lot to Sytze de Bruin from Wageningen University for a lot of help with the geostatistics.

[^1]: G. B. M. Heuvelink, E. Pebesma, and B. Gräler, “Space-Time Geostatistics,” in Encyclopedia of GIS, S. Shekhar, H. Xiong, and X. Zhou, Eds., Cham: Springer International Publishing, 2017, pp. 1919–1926. doi: 10.1007/978-3-319-17885-1_1647.

[^2]: B. Graeler, E. Pebesma, and G. Heuvelink, “Spatio-Temporal Interpolation using gstat,” The R Journal, vol. 8, pp. 204–218, Jan. 2016, doi: 10.32614/RJ-2016-014.

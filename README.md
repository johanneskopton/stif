# STIF

[![PyPI - Version](https://img.shields.io/pypi/v/stif)](https://pypi.org/project/stif/)
[![Test](https://johannes.kopton.org/stif/_static/tests-badge.svg)](https://johannes.kopton.org/stif/_static/reports/junit/report.html)
[![Test coverage](https://johannes.kopton.org/stif/_static/coverage-badge.svg)](https://johannes.kopton.org/stif/_static/reports/htmlcov/index.html)
[![Flake8](https://johannes.kopton.org/stif/_static/flake8-badge.svg)](https://johannes.kopton.org/stif/_static/reports/flake8)

"Space-time interpolation and forecasting"

Package for predicting spatio-temporally distributed variables via *space-time regression Kriging*, using numpy and numba.

## Resources

* [Introduction Notebook](https://github.com/johanneskopton/stif/blob/main/docs/introduction.ipynb) <a target="_blank" href="https://colab.research.google.com/github/johanneskopton/stif/blob/main/docs/introduction.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [Documentation and API reference](https://johannes.kopton.org/stif)
* [Issue tracker](https://github.com/johanneskopton/stif/issues)

What you can do with this package:

* create space-time variograms (even on very large datasets)
* fit space-time variogram models (e.g. using `product_sum` or `sum_metric` models)
* fit regression models on covariates
* perform space-time variogram Kriging to interpolate/extrapolate from the input data
* do all this with binary (presence/absence) indicators, e.g. for dynamic species distribution modeling

## Examples

![Example variogram models](https://raw.githubusercontent.com/johanneskopton/stif/main/docs/source/_static/demo_variogram.png)

![Example PM10 predictions for 2015-03-01 for Germany](https://raw.githubusercontent.com/johanneskopton/stif/main/docs/source/_static/demo_map.png)

## Space-time regression Kriging
Using this package, you can do _space-time regression Kriging_. But what is it exactly? Imagine you have measurements on different days and distributed over a geographical region (basically a table with latitude, longitude, time and the measured variable of interest). And now you want to estimate the value of this variable at locations and dates, at which you didn't measure. This can be the future (forecasting) or just some day in between, where no measurement was taken (interpolation). You may have some other variables (external covariates), that can explain some of the variation of you variable of interest via some form of regression. In this case, it makes sense to do the Kriging only on the residuals of this regression (i.e. *regression Kriging*). All of this, you can do using this package (and hopefully as simple as possible).


## Why this package?
To my knowledge, there is no other Python package for space-time variogram Kriging on unstructured spatio-temporal data. While the R package [gstat](http://r-spatial.github.io/gstat/) [^1] can do space-time Kriging, there can be problems with large datasets. Furthermore, the R package obviously doesn't play well with the Python ecosystem, e.g. using Keras/tensorflow for the regression part of regression Kriging.

There is the excellent [scikit-gstat](https://github.com/mmaelicke/scikit-gstat) [^2], but as for now it can not create variograms for unstructured spatio-temporal data and I didn't have the time yet to dive into it and add this feature. So perhaps for now, you may find this package useful.

You can also do something like space-time regression Kriging using one of the more mature Gaussian process libraries out there (since Kriging is actually just Gaussian process regression), but they won't provide you with nice variograms and all the other geostats stuff.

<!--
<center><img src="https://raw.githubusercontent.com/johanneskopton/stif/main/docs/source/_static/demo1.gif" alt="Example: PM10 values in Germany" width="500"/></center>
-->
## Installation

Via pip:

```sh
pip install .                   # for the core package
pip install .[tensorflow]       # for using tensorflow/keras models for covariate regression
pip install .[geo]              # for using coordinate transformations and geo I/O
pip install .[interactive]      # for running the example jupyter notebooks
pip install .[dev]              # for all the development tools used in this project
```

## Development

Installing via pip for development:

```sh
pip install -e .[dev]
```

Testing:

```sh
pytest --cov=stif --junitxml=docs/source/_static/reports/junit/junit.xml --html=docs/source/_static/reports/junit/report.html
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

Build package:
```sh
python -m build
```

Publish package on PyPI:
```sh
twine upload dist/*
```

## Acknowledgements
Heavily inspired by [^3] and the space-time variogram implementation in the R package [gstat](http://r-spatial.github.io/gstat/). Thanks a lot to Sytze de Bruin from Wageningen University for a lot of help with the geostatistics.

[^1]: B. Graeler, E. Pebesma, and G. Heuvelink, “Spatio-Temporal Interpolation using gstat,” The R Journal, vol. 8, pp. 204–218, Jan. 2016, doi: 10.32614/RJ-2016-014.

[^2]: M. Mälicke, “SciKit-GStat 1.0: a SciPy-flavored geostatistical variogram estimation toolbox written in Python,” Geoscientific Model Development, vol. 15, no. 6, pp. 2505–2532, Mar. 2022, doi: 10.5194/gmd-15-2505-2022.

[^3]: G. B. M. Heuvelink, E. Pebesma, and B. Gräler, “Space-Time Geostatistics,” in Encyclopedia of GIS, S. Shekhar, H. Xiong, and X. Zhou, Eds., Cham: Springer International Publishing, 2017, pp. 1919–1926. doi: 10.1007/978-3-319-17885-1_1647.

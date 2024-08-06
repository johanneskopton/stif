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
genbadge tests -i docs/source/_static/reports/junit/junit.xml -o docs/source/_static/tests-badge.svg
genbadge coverage -i docs/source/_static/reports/coverage.xml -o docs/source/_static/coverage-badge.svg
```

Build sphinx documentation:
```sh
sphinx-build -M html docs/source/ docs/build/
```

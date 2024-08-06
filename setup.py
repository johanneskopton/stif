import os

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="fist",
    version="0.0.1",
    author="Johannes Kopton",
    author_email="johannes@kopton.org",
    description=("Forecasting and Interpolation in Space and Time"),
    keywords="kriging variogram spacetime spatio-temporal",
    url="https://github.com/johanneskopton/fist",
    packages=["fist"],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3"
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "geopandas",
        "scikit-learn",
        "numba",
        "scipy",
    ],
    extras_require={
        "tensorflow": [
            "tensorflow",
            "keras",
        ],
        "dev": [
            "pytest",
            "pytest-cov",
            "pytest-html",
            "genbadge[all]",
            "flake8",
            "autopep8",
            "pre-commit",
            "sphinx",
            "myst-parser",
            "pydata-sphinx-theme",
        ],
    },
)

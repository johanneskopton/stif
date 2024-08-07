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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Framework :: tox",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.19,<2.0",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "numba>=0.54,<=0.60",
        "scipy",
    ],
    extras_require={
        "tensorflow": [
            "tensorflow",
            "keras",
        ],
        "geo": [
            "geopandas",
            "shapely",
            "rasterio",
        ],
        "interactive": [
            "ipykernel",
        ],
        "dev": [
            "pytest",
            "pytest-cov",
            "pytest-html",
            "genbadge[all]",
            "flake8",
            "flake8-html",
            "autopep8",
            "pre-commit",
            "sphinx",
            "myst-parser",
            "pydata-sphinx-theme",
        ],
    },
)

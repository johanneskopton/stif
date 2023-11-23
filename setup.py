import os

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="minkowski",
    version="0.0.1",
    author="Johannes Kopton",
    author_email="johannes@kopton.org",
    description=("Predicting the probability of species minkowskis"),
    keywords="kriging species distribution spacetime spatio-temporal",
    url="https://github.com/johanneskopton/minkowski",
    packages=["minkowski"],
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
        "dev": [
            "pytest",
            "flake8",
            "autopep8",
            "pre-commit",
        ],
    },
)

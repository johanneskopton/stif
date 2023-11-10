import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="occurence",
    version="0.0.1",
    author="Johannes Kopton",
    author_email="johannes@kopton.org",
    description=("Predicting the probability of species occurences"),
    keywords="kriging species distribution spacetime spatio-temporal",
    url="https://github.com/johanneskopton/occurence",
    packages=["occurence"],
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
        "geopandas"
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "autopep8"
        ],
    },
)

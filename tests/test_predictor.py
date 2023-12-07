import copy
import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .read_pm10_test_data import df
from .read_pm10_test_data import df_binary
from ffist import Data
from ffist import Predictor
from ffist import sinusodial_feature_transform


def test_covariance_predictor_binary():
    data = Data(
        df_binary,
        space_cols=["x", "y"],
        time_col="time",
        covariate_cols=["x", "y", "time"],
    )

    models = [LogisticRegression, RandomForestClassifier]
    target_aucs = [0.60, 0.64]

    for i, model in enumerate(models):
        covariate_model = model(random_state=0)
        predictor = Predictor(data, covariate_model)
        cv_aucs = predictor.get_cross_val_metric(sklearn.metrics.roc_auc_score)
        assert np.isclose(cv_aucs[-1], target_aucs[i], rtol=0.1)


def test_sinusodials():
    n = 10000
    time = np.random.uniform(0, 1000, n)
    data = {
        "x": np.random.uniform(0, 10, n),
        "y": np.random.uniform(0, 10, n),
        "time": time,
        "predictand": (time*0.001 + np.sin(time/365*2*math.pi)) > 0.5,
    }
    d = pd.DataFrame(data)

    data = Data(
        d,
        space_cols=["x", "y"],
        time_col="time",
        covariate_cols=["x", "y", "time"],
        covariate_transformations={
            "x": sinusodial_feature_transform,
            "y": sinusodial_feature_transform,
            "time": lambda x: sinusodial_feature_transform(
                x,
                n_freqs=5,
                full_circle=365,
            ),
        },
    )
    covariate_model = MLPClassifier(
        hidden_layer_sizes=[2],
        random_state=0,
        max_iter=1000,
    )
    predictor = Predictor(
        data,
        cv_splits=3,
        covariate_model=covariate_model,
    )
    cv_aucs = predictor.get_cross_val_metric(sklearn.metrics.roc_auc_score)
    assert np.allclose(cv_aucs, [1, 1, 1], rtol=0.02)


def test_covariance_predictor_transformation_binary():
    data = Data(
        df_binary,
        space_cols=["x", "y"],
        time_col="time",
        covariate_cols=["x", "y", "time"],
        covariate_transformations={
            "x": sinusodial_feature_transform,
            "y": sinusodial_feature_transform,
            "time": lambda x: sinusodial_feature_transform(
                x,
                n_freqs=2,
                full_circle=365,
            ),
        },
    )
    covariate_model = MLPClassifier(
        hidden_layer_sizes=[3],
        random_state=0,
        max_iter=500,
    )
    predictor = Predictor(
        data,
        cv_splits=3,
        covariate_model=covariate_model,
    )
    cv_aucs = predictor.get_cross_val_metric(sklearn.metrics.roc_auc_score)
    predictor.plot_cross_validation_roc(
        target=tempfile.NamedTemporaryFile(delete=True),
    )

    assert np.allclose(cv_aucs, [0.6, 0.5, 0.66], rtol=0.1)


def test_residuals_binary():
    data = Data(
        df_binary,
        space_cols=["x", "y"],
        time_col="time",
        covariate_cols=["x", "y", "time"],
    )

    covariate_model = LogisticRegression(random_state=0)
    predictor = Predictor(data, covariate_model)
    predictor.fit_covariate_model()
    residuals = predictor.get_residuals()
    assert np.isclose(residuals.mean(), 0, atol=1e-4)
    assert np.isclose(residuals.std(), 0.46, rtol=0.1)


def test_save_covariate_model_sklearn():
    data = Data(
        df_binary,
        space_cols=["x", "y"],
        time_col="time",
        covariate_cols=["x", "y", "time"],
    )

    covariate_model = LogisticRegression(random_state=0)
    predictor1 = Predictor(data, covariate_model)
    predictor2 = Predictor(data, None)

    predictor1.fit_covariate_model()
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        predictor1.save_covariate_model(temp_filename)
        predictor2.load_covariate_model(temp_filename)
    os.remove(temp_filename)

    residuals1 = predictor1.get_residuals()
    residuals2 = predictor2.get_residuals()

    assert np.allclose(residuals1, residuals2, rtol=0.1)


def test_empirical_variogram():
    data = Data(
        df,
        space_cols=["x", "y"],
        time_col="time",
        predictand_col="PM10",
        covariate_cols=["x", "y", "time"],
    )

    covariate_model = LinearRegression()
    predictor = Predictor(data, covariate_model)
    predictor.fit_covariate_model()

    predictor.calc_empirical_variogram(
        space_dist_max=6e5,
        time_dist_max=10,
        el_max=1e8,
    )

    assert np.isclose(predictor._variogram.min(), 27.8, rtol=0.1)
    assert np.isclose(predictor._variogram.max(), 136.0, rtol=0.1)


def test_save_empirical_variogram():
    data = Data(
        df,
        space_cols=["x", "y"],
        time_col="time",
        predictand_col="PM10",
        covariate_cols=["x", "y", "time"],
    )

    covariate_model = LinearRegression()
    predictor1 = Predictor(data, covariate_model)
    predictor1.fit_covariate_model()

    predictor2 = copy.deepcopy(predictor1)

    predictor1.calc_empirical_variogram(
        space_dist_max=6e5,
        time_dist_max=10,
        el_max=1e8,
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as temp_file:
        temp_filename = temp_file.name
        predictor1.save_empirical_variogram(temp_filename)
        predictor2.load_empirical_variogram(temp_filename)

    variogram1 = predictor1._variogram
    variogram2 = predictor2._variogram

    assert np.allclose(variogram1, variogram2, rtol=0.1)


def test_fit_variogram_model():
    data = Data(
        df,
        space_cols=["x", "y"],
        time_col="time",
        predictand_col="PM10",
        covariate_cols=["x", "y", "time"],
    )

    covariate_model = LinearRegression()
    predictor = Predictor(data, covariate_model)
    predictor.fit_covariate_model()

    predictor.calc_empirical_variogram(
        space_dist_max=6e5,
        time_dist_max=7,
        n_time_bins=7,
        el_max=1e8,
    )

    predictor.fit_variogram_model()
    predictor.plot_variogram_model_comparison(
        target=tempfile.NamedTemporaryFile(delete=True),
    )

    assert np.isclose(5.36, predictor._variogram_fit.fun, rtol=0.2)


@pytest.mark.filterwarnings("ignore::RuntimeWarning:matplotlib")
def test_kriging():
    data = Data(
        df,
        space_cols=["x", "y"],
        time_col="time",
        predictand_col="PM10",
        covariate_cols=["x", "y", "time"],
    )

    covariate_model = LinearRegression()
    predictor = Predictor(data, covariate_model)
    predictor.fit_covariate_model()

    predictor.calc_empirical_variogram(
        space_dist_max=6e5,
        time_dist_max=7,
        n_time_bins=7,
        el_max=1e8,
    )

    predictor.fit_variogram_model()
    sample_pos = df.iloc[21000:21025]
    space = sample_pos[["x", "y"]].to_numpy()
    time = sample_pos["time"].to_numpy()

    predictor.plot_kriging_weights(
        space[0, :], time[0],
        max_kriging_points=50,
        target=tempfile.NamedTemporaryFile(delete=True),
    )
    kriging_mean, kriging_std = predictor.get_kriging_prediction(
        space, time,
    )
    assert np.isclose(kriging_mean[0], 0.888, atol=2.0)

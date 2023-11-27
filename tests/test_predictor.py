import tempfile

import numpy as np
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .read_pm10_test_data import df
from .read_pm10_test_data import df_binary
from minkowski import Data
from minkowski import Predictor
from minkowski import sinusodial_feature_transform


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
                n_freqs=5,
                full_circle=365,
            ),
        },
    )
    covariate_model = MLPClassifier(
        hidden_layer_sizes=[5],
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

    assert np.isclose(cv_aucs, [0.7, 0.53, 0.7], rtol=0.1).all()


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

import numpy as np
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .read_pm10_test_data import df
from occurence import OccurenceData
from occurence import Predictor
from occurence import sinusodial_feature_transform


def test_init_covariance_predictor():
    data = OccurenceData(
        df,
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


def test_init_covariance_predictor_transformation():
    data = OccurenceData(
        df,
        space_cols=["x", "y"],
        time_col="time",
        covariate_cols=["x", "y", "time"],
        covariate_transformations={
            "x": sinusodial_feature_transform,
            "y": sinusodial_feature_transform,
            "time": lambda x: sinusodial_feature_transform(x, 5),
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

    assert np.isclose(cv_aucs, [0.7, 0.53, 0.7], rtol=0.1).all()


def test_residuals():
    data = OccurenceData(
        df,
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

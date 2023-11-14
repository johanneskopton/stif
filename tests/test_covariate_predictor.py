import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .read_pm10_test_data import df
from occurence import CovariancePredictor
from occurence import OccurenceData
from occurence import sinusodial_feature_transform


def test_init_covariance_predictor():
    data = OccurenceData(df, space_cols=["x", "y"], time_col="time")

    models = [LogisticRegression, RandomForestClassifier]
    target_aucs = [0.60, 0.66]

    for i, model in enumerate(models):
        predictor = CovariancePredictor(
            data,
            model,
            ["x", "y", "time"],
            model_params={
                "random_state": 0,
            },
        )
        predictor.fit()
        assert np.isclose(predictor.roc_auc_score, target_aucs[i], rtol=0.1)


def test_init_covariance_predictor_transformation():
    data = OccurenceData(df, space_cols=["x", "y"], time_col="time")

    predictor = CovariancePredictor(
        data,
        MLPClassifier,
        ["x", "y", "time"],
        covariate_transformations={
            "x": sinusodial_feature_transform,
            "y": sinusodial_feature_transform,
            "time": lambda x: sinusodial_feature_transform(x, 5),
        },
        model_params={
            "hidden_layer_sizes": [5],
            "random_state": 0,
            "max_iter": 500,
        },
    )
    predictor.fit()
    assert np.isclose(predictor.roc_auc_score, 0.75, rtol=0.1)

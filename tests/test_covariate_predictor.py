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
    data = OccurenceData(df, space_cols=["x", "y"], time_col="time")

    models = [LogisticRegression, RandomForestClassifier]
    target_aucs = [0.60, 0.64]

    for i, model in enumerate(models):
        classifier = model(random_state=0)
        predictor = Predictor(
            data,
        )
        predictor.init_covariate_model(
            classifier,
            ["x", "y", "time"],
        )
        predictor.fit_covariate_model()
        cv_aucs = predictor.get_cross_val_metric(sklearn.metrics.roc_auc_score)
        assert np.isclose(cv_aucs[-1], target_aucs[i], rtol=0.1)


def test_init_covariance_predictor_transformation():
    data = OccurenceData(df, space_cols=["x", "y"], time_col="time")

    predictor = Predictor(
        data,
        cv_splits=3,
    )
    classifier = MLPClassifier(
        hidden_layer_sizes=[5],
        random_state=0,
        max_iter=500,
    )
    predictor.init_covariate_model(
        classifier,
        ["x", "y", "time"],
        covariate_transformations={
            "x": sinusodial_feature_transform,
            "y": sinusodial_feature_transform,
            "time": lambda x: sinusodial_feature_transform(x, 5),
        },
    )
    predictor.fit_covariate_model()
    cv_aucs = predictor.get_cross_val_metric(sklearn.metrics.roc_auc_score)

    assert np.isclose(cv_aucs, [0.7, 0.53, 0.7], rtol=0.1).all()

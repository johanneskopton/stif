import numpy as np

from .read_pm10_test_data import df
from occurence import OccurenceData


def test_init_occurence_data():
    occuence_data = OccurenceData(df, space_cols=["x", "y"], time_col="time")
    assert np.isclose(
        occuence_data.space_coords.mean(
            axis=0,
        ), [598158, 5685813], rtol=0.1,
    ).all()
    assert np.isclose(occuence_data.time_coords.mean(), 5294, rtol=0.1)


def test_covariate_normalization():
    df["cov"] = np.random.randint(4, 17, len(df))
    occuence_data = OccurenceData(
        df,
        space_cols=["x", "y"],
        time_col="time",
        covariate_cols=["cov"],
    )
    X = occuence_data.get_training_covariates()
    assert (X.min(axis=0) >= 0).all()
    assert (X.max(axis=0) <= 1).all()

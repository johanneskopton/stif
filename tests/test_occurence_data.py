import numpy as np

from .read_pm10_test_data import df
from occurence import OccurenceData


def test_init_occurence_data():
    occuence_data = OccurenceData(df, space_cols=["x", "y"], time_col="time")
    assert np.isclose(
        occuence_data.space_coords.mean(
            axis=0,
        ), [0.48, 0.49], rtol=0.1,
    ).all()
    assert np.isclose(occuence_data.time_coords.mean(), 0.49, rtol=0.1)

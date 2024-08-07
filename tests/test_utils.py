import numpy as np

import stif.utils


def test_distance_matrix_1d():
    vec = np.array([1, 2, 3, 4, 5])
    res = stif.utils.calc_distance_matrix_1d(vec)
    assert np.allclose(
        res,
        np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0],
            ],
        ),
    )


def test_distance_matrix_2d():
    vec = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    res = stif.utils.calc_distance_matrix_2d(vec)
    assert np.allclose(
        res,
        np.linalg.norm(vec[:, None, :] - vec[None, :, :], axis=-1),
    )

import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def calc_distance_matrix_1d(vec):
    """Calculate full distance matrix for 1D vector.

    Parameters
    ----------
    vec : Numpy array of shape (n,)
        Input vector

    Returns
    -------
    NumPy array of shape (n, n)
        Output matrix
    """
    res = np.empty((len(vec), len(vec)), dtype=vec.dtype)
    for i in range(len(vec)):
        for j in range(len(vec)):
            res[i, j] = np.abs(vec[i]-vec[j])
    return res


@nb.njit(fastmath=True)
def calc_distance_matrix_2d(vec):
    """Calculate full distance matrix using euclidean distance.

    Parameters
    ----------
    vec : Numpy array of shape (n, 2)
        Input vector

    Returns
    -------
    NumPy array of shape (n, n)
        Output matrix
    """
    res = np.empty((vec.shape[0], vec.shape[0]), dtype=vec.dtype)
    for i in range(vec.shape[0]):
        for j in range(vec.shape[0]):
            res[i, j] = (vec[i, 0]-vec[j, 0]) ** 2 + (vec[i, 1]-vec[j, 1])**2
    return np.sqrt(res)


# @nb.njit(fastmath=True)
# def cosine_distance(x, y):
#     """Calculate cosine distance between two vectors.

#     Parameters
#     ----------
#     x : Numpy array of shape (n,)
#         Input vector
#     y : Numpy array of shape (n,)
#         Input vector

#     Returns
#     -------
#     float
#         Cosine distance
#     """
#     return (1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))) / 2


# @nb.njit(fastmath=True)
# def calc_distance_matrix_cosine(vec):
#     """Calculate full distance matrix using cosine distance.

#     Parameters
#     ----------
#     vec : Numpy array of shape (n, m)
#         Input vector

#     Returns
#     -------
#     Numpy array of shape (n, n)
#         Output matrix
#     """
#     res = np.empty((vec.shape[0], vec.shape[0]), dtype=vec.dtype)
#     for i in range(vec.shape[0]):
#         for j in range(vec.shape[0]):
#             res[i, j] = cosine_distance(vec[i, :], vec[j, :])
#     return res


@nb.njit(fastmath=True)
def _pair_index_generator(n, n_samples=None):
    if n_samples is None:
        for i in range(n):
            for j in range(i+1, n):
                yield i, j
    else:
        for _ in range(n_samples):
            i = np.random.randint(n)
            j = np.random.randint(n)

            if i == j:
                continue

            yield i, j


@nb.njit(fastmath=True)
def get_variogram(
    space,
    time,
    val,
    space_dist_max,
    time_dist_max,
    n_space_bins,
    n_time_bins,
    n_samples,
):
    """Calculate empirical variogram.

    Parameters
    ----------
    space : Numpy array of shape (n, 2)
        Spatial coordinates
    time : Numpy array of shape (n,)
        Temporal coordinates
    val : Numpy array of shape (n,)
        Values
    space_dist_max : float
        Maximum spatial distance
    time_dist_max : float
        Maximum temporal distance
    n_space_bins : int
        Number of spatial bins
    n_time_bins : int
        Number of temporal bins
    n_samples : int
        Maximum number of samples

    Returns
    -------
    tuple
        Variogram (numpy array of shape (n_space_bins, n_time_bins)),
        norm (numpy array of shape (n_space_bins, n_time_bins)),
        space_bin_width (float), time_bin_width (float)
    """
    n = len(val)

    # prepare histogram
    space_bin_width = space_dist_max / n_space_bins
    time_bin_width = time_dist_max / n_time_bins
    hist = np.zeros((n_space_bins, n_time_bins), dtype=np.float64)
    norm = np.zeros((n_space_bins, n_time_bins), dtype=np.float64)

    for i, j in _pair_index_generator(n, n_samples):
        # if distance == "euclidean":
        space_lag = np.sqrt(
            np.square(space[i, 0]-space[j, 0]) +
            np.square(space[i, 1]-space[j, 1]),
        )
        # elif distance == "cosine":
        #     space_lag = cosine_distance(space[i, :], space[j, :])

        if space_lag > space_dist_max:
            continue
        time_lag = np.abs(time[i]-time[j])
        if time_lag > time_dist_max:
            continue
        sq_val_delta = np.square(val[i]-val[j])

        # space_lag, time_lag, sq_val_delta
        space_bin = int(space_lag / space_bin_width)
        time_bin = int(time_lag / time_bin_width)

        if 0 <= space_bin < n_space_bins and\
                0 <= time_bin < n_time_bins:
            hist[space_bin, time_bin] += sq_val_delta
            norm[space_bin, time_bin] += 1

    # I think this "/2" is necessary, because in samples_per_bin are only
    # n^2/2 samples in total
    variogram = np.divide(
        hist,
        norm,
        # out=np.ones_like(hist) * np.nan,
        # where=norm != 0,
    ) / 2

    for i in range(n_space_bins):
        for j in range(n_time_bins):
            if norm[i, j] == 0:
                variogram[i, j] = np.nan

    return variogram, norm, space_bin_width, time_bin_width


@nb.njit(fastmath=True)
def get_covariogram(
    features,
    time,
    val,
    space_dist_max,
    time_dist_max,
    n_space_bins,
    n_time_bins,
    n_samples,
):
    """Calculate empirical covariogram.

    Parameters
    ----------
    space : Numpy array of shape (n, 2)
        Spatial coordinates
    time : Numpy array of shape (n,)
        Temporal coordinates
    val : Numpy array of shape (n,)
        Values
    space_dist_max : float
        Maximum spatial distance
    time_dist_max : float
        Maximum temporal distance
    n_space_bins : int
        Number of spatial bins
    n_time_bins : int
        Number of temporal bins
    n_samples : int
        Maximum number of samples

    Returns
    -------
    tuple
        Covariogram (numpy array of shape (n_space_bins, n_time_bins)),
        norm (numpy array of shape (n_space_bins, n_time_bins)),
        space_bin_width (float), time_bin_width (float)
    """
    n = len(val)

    mean = np.mean(val)
    var = np.var(val)

    # prepare histogram
    space_bin_width = space_dist_max / n_space_bins
    time_bin_width = time_dist_max / n_time_bins
    hist = np.zeros((n_space_bins, n_time_bins), dtype=np.float64)
    norm = np.zeros((n_space_bins, n_time_bins), dtype=np.float64)

    for i, j in _pair_index_generator(n, n_samples):
        features_lag = np.sqrt(
            np.square(features[i, 0]-features[j, 0]) +
            np.square(features[i, 1]-features[j, 1]),
        )

        if features_lag > space_dist_max:
            continue
        time_lag = np.abs(time[i]-time[j])
        if time_lag > time_dist_max:
            continue
        sq_val_delta = (val[i]-mean)*(val[j]-mean)

        # space_lag, time_lag, sq_val_delta
        space_bin = int(features_lag / space_bin_width)
        time_bin = int(time_lag / time_bin_width)

        if 0 <= space_bin < n_space_bins and\
                0 <= time_bin < n_time_bins:
            hist[space_bin, time_bin] += sq_val_delta
            norm[space_bin, time_bin] += 1

    # I think this "/2" is necessary, because in samples_per_bin are only
    # n^2/2 samples in total
    covariogram = np.divide(
        hist,
        norm,
        # out=np.ones_like(hist) * np.nan,
        # where=norm != 0,
    ) / var

    for i in range(n_space_bins):
        for j in range(n_time_bins):
            if norm[i, j] == 0:
                covariogram[i, j] = np.nan

    return covariogram, norm, space_bin_width, time_bin_width

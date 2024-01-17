import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def calc_distance_matrix_1d(vec):
    res = np.empty((len(vec), len(vec)), dtype=vec.dtype)
    for i in range(len(vec)):
        for j in range(len(vec)):
            res[i, j] = np.abs(vec[i]-vec[j])
    return res


@nb.njit(fastmath=True)
def calc_distance_matrix_2d(vec):
    res = np.empty((vec.shape[0], vec.shape[0]), dtype=vec.dtype)
    for i in range(vec.shape[0]):
        for j in range(vec.shape[0]):
            res[i, j] = np.sqrt(
                (vec[i, 0]-vec[j, 0])**2 +
                (vec[i, 1]-vec[j, 1])**2,
            )
    return res


@nb.njit(fastmath=True)
def pair_index_generator(n, n_samples=None):
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
    n = len(val)

    # prepare histogram
    space_bin_width = space_dist_max / n_space_bins
    time_bin_width = time_dist_max / n_time_bins
    hist = np.zeros((n_space_bins, n_time_bins), dtype=np.float64)
    norm = np.zeros((n_space_bins, n_time_bins), dtype=np.float64)

    for i, j in pair_index_generator(n, n_samples):
        space_lag = np.sqrt(
            np.square(space[i, 0]-space[j, 0]) +
            np.square(space[i, 1]-space[j, 1]),
        )
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

import math

import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def spherical(h, r, c0, b=0):
    a = r / 1.
    if h <= r:
        return b + c0 * ((1.5 * (h / a)) - (0.5 * ((h / a) ** 3.0)))
    else:
        return b + c0


@nb.njit(fastmath=True)
def gaussian(h, r, c0, b=0):
    a = r / 2.
    return b + c0 * (1. - math.exp(- (h ** 2 / a ** 2)))


@nb.njit(fastmath=True)
def sum_model(h, t, x, models):
    r_s, r_t, c0_s, c0_t, b_s, b_t = x[0], x[1], x[2], x[3], x[4], x[5]
    return models[0](h, r_s, c0_s, b_s) + models[1](t, r_t, c0_t, b_t)


@nb.njit(fastmath=True)
def product_model(h, t, x, models):
    r_s, r_t, c0_s, c0_t, b_s, b_t = x[0], x[1], x[2], x[3], x[4], x[5]
    return models[0](h, r_s, c0_s, b_s) * models[1](t, r_t, c0_t, b_t)


@nb.njit(fastmath=True)
def product_sum_model(h, t, x, models):
    k = x[6]
    return k * product_model(h, t, x[:-1], models)\
        + product_model(h, t, x[:-1], models)


@nb.njit(fastmath=True)
def metric_model(h, t, x, models):
    r, c0, b, ani = x[0], x[1], x[2], x[3]
    metric_dist = math.sqrt(h*h + ani*ani*t*t)
    return models[2](metric_dist, r, c0, b)


@nb.njit(fastmath=True)
def sum_metric_model(h, t, x, models):
    r_s, r_t, r_m, c0_s, c0_t, c0_m, b_s, b_t, b_m, ani = \
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]
    metric_dist = math.sqrt(h*h + ani*ani*t*t)
    return sum_model(h, t, [r_s, r_t, c0_s, c0_t, b_s, b_t], models)\
        + models[2](metric_dist, r_m, c0_m, b_m)


@nb.njit()
def prediction_grid(
    x,
    st_model,
    models,
    bins_spatial,
    bins_temporal,
):
    variogram = np.empty(
        (len(bins_spatial), len(bins_temporal)), dtype=np.double,
    )
    for i, h in enumerate(bins_spatial):
        for j, t in enumerate(bins_temporal):
            variogram[i, j] = st_model(h, t, x, models)
    return variogram


def weighted_mean_square_error(
    x,
    st_model,
    models,
    empirical_variogram,
    weights,
):
    n_space_bins, n_time_bins = empirical_variogram.shape
    prediction = prediction_grid(
        x,
        st_model, models,
        n_space_bins, n_time_bins,
    )
    error_grid = empirical_variogram - prediction
    return np.average(np.square(error_grid), weights=weights)


def calc_weights(space_bins, time_bins, ani, samples_per_bin):
    n_space_bins = len(space_bins)
    n_time_bins = len(time_bins)

    distances_sq = np.square(ani * np.tile(time_bins, [n_space_bins, 1]))\
        + np.square(np.tile(np.expand_dims(space_bins, 0).T, [1, n_time_bins]))

    # looks better but without sqrt in literature
    weights = samples_per_bin.astype(float) / np.sqrt(distances_sq)

    # normalize weights
    weights /= np.max(weights)

    return weights


def get_initial_parameters(
    model_str, empirical_variogram, space_dist_max, time_dist_max, ani,
):
    r_s = space_dist_max  # range space
    r_t = time_dist_max  # range time
    r_m = space_dist_max  # range metric

    c0_s = empirical_variogram[:, 0].max()  # sill space
    c0_t = empirical_variogram[0, :].max() - c0_s  # sill time
    c0_m = empirical_variogram.max()  # sill metric

    b = empirical_variogram.min() / 2  # half nugget

    initial_params_dict = {
        "sum": [r_s, r_t, c0_s, c0_t, b, b],
        "product": [r_s, r_t, c0_s, c0_t, b, b],
        "product_sum": [r_s, r_t, c0_s/2, c0_t/2, b/2, b/2, 0.5],
        "metric": [r_m, c0_m, b*2, ani],
        "sum_metric": [
            r_s, r_t, r_m, c0_s/2, c0_t/2, c0_m/2, b/2, b/2, b, ani,
        ],
    }

    return np.array(initial_params_dict[model_str])


variogram_model_dict = {
    "spherical": spherical,
    "gaussian": gaussian,
    "sum": sum_model,
    "product": product_model,
    "product_sum": product_sum_model,
    "metric": metric_model,
    "sum_metric": sum_metric_model,
}

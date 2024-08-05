import math

import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def spherical(h, r, c0, b=0.0):
    """Spherical variogram model.

    .. math::

        \\gamma = b + C_0 * \\left({1.5*\\frac{h}{r} - 0.5*\\frac{h}{r}^3}
        \\right)

    if h<r, and

    .. math::

        \\gamma = b + C_0

    otherwise.

    Parameters
    ----------
    h : float
        Lag distance.
    r : float
        Effective range.
    c0 : float
        Sill.
    b : float, optional
        Nugget, by default 0.0

    Returns
    -------
    float
        Variogram value.
    """
    a = r / 1.
    if h <= r:
        return b + c0 * ((1.5 * (h / a)) - (0.5 * ((h / a) ** 3.0)))
    else:
        return b + c0


@nb.njit(fastmath=True)
def gaussian(h, r, c0, b=0.0):
    """Gaussian variogram model.

    .. math::
        \\gamma = b + c_0 * \\left({1 - e^{-\\frac{4 \\cdot h^2}{r^2}}}\\right)

    Parameters
    ----------
    h : float
        Lag distance.
    r : float
        Effective range.
    c0 : float
        Sill.
    b : float, optional
        Nugget, by default 0.0

    Returns
    -------
    float
        Variogram value.
    """
    a = r / 2.
    return b + c0 * (1. - math.exp(- (h ** 2 / a ** 2)))


@nb.njit(fastmath=True)
def sum_model(h, t, x, model_space, model_time, model_metric):
    """Sum model for combining the spatial and temporal components.

    Parameters
    ----------
    h : float
        Space lag distance.
    t : float
        Time lag distance.
    x : tuple
        Parameters for the spatial and temporal models.
    model_space : function
        Spatial model.
    model_time : function
        Temporal model.
    model_metric : function
        Metric model.

    Returns
    -------
    float
        Predicted semivariance.
    """
    r_s, r_t, c0_s, c0_t, b_s, b_t = x[0], x[1], x[2], x[3], x[4], x[5]
    return model_space(h, r_s, c0_s, b_s) + model_time(t, r_t, c0_t, b_t)


@nb.njit(fastmath=True)
def product_model(h, t, x, model_space, model_time, model_metric):
    """Product model for combining the spatial and temporal components.

    Parameters
    ----------
    h : float
        Space lag distance.
    t : float
        Time lag distance.
    x : tuple
        Parameters for the spatial and temporal models.
    model_space : function
        Spatial model.
    model_time : function
        Temporal model.
    model_metric : function
        Metric model.

    Returns
    -------
    float
        Predicted semivariance.
    """
    r_s, r_t, c0_s, c0_t, b_s, b_t = x[0], x[1], x[2], x[3], x[4], x[5]
    return model_space(h, r_s, c0_s, b_s) * model_time(t, r_t, c0_t, b_t)


@nb.njit(fastmath=True)
def product_sum_model(h, t, x, model_space, model_time, model_metric):
    """Product-sum model for combining the spatial and temporal components.

    Parameters
    ----------
    h : float
        Space lag distance.
    t : float
        Time lag distance.
    x : tuple
        Parameters for the spatial and temporal models plus weighting
        coefficient.
    model_space : function
        Spatial model.
    model_time : function
        Temporal model.
    model_metric : function
        Metric model.

    Returns
    -------
    float
        Predicted semivariance.
    """
    k = x[6]
    return k * product_model(h, t, x[:-1], model_space, model_time, None)\
        + product_model(h, t, x[:-1], model_space, model_time, None)


@nb.njit(fastmath=True)
def metric_model(h, t, x, model_space, model_time, model_metric):
    """Metric model for combining the spatial and temporal components.

    Parameters
    ----------
    h : float
        Space lag distance.
    t : float
        Time lag distance.
    x : tuple
        Parameters for the metric model, including anisotropy.
    model_space : function
        Spatial model.
    model_time : function
        Temporal model.
    model_metric : function
        Metric model.

    Returns
    -------
    float
        Predicted semivariance.
    """
    r, c0, b, ani = x[0], x[1], x[2], x[3]
    metric_dist = math.sqrt(h*h + ani*ani*t*t)
    return model_metric(metric_dist, r, c0, b)


@nb.njit(fastmath=True)
def sum_metric_model(h, t, x, model_space, model_time, model_metric):
    """Sum-metric model for combining the spatial and temporal components.

    Parameters
    ----------
    h : float
        Space lag distance.
    t : float
        Time lag distance.
    x : tuple
        Parameters for the spatial, temporal and metric model.
    model_space : function
        Spatial model.
    model_time : function
        Temporal model.
    model_metric : function
        Metric model.

    Returns
    -------
    float
        Predicted semivariance.
    """
    r_s, r_t, r_m, c0_s, c0_t, c0_m, b_s, b_t, b_m, ani = \
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]
    metric_dist = math.sqrt(h*h + ani*ani*t*t)
    x_sum = [r_s, r_t, c0_s, c0_t, b_s, b_t]
    return sum_model(h, t, x_sum, model_space, model_time, None)\
        + model_metric(metric_dist, r_m, c0_m, b_m)


@nb.njit()
def prediction_grid(
    x,
    st_model,
    model_space, model_time, model_metric,
    bins_space,
    bins_time,
):
    """Calculate the variogram model results on a given grid.

    Parameters
    ----------
    x : tuple
        Variogram model parameters.
    st_model : function
        Spate-time variogram model function.
    model_space : function
        Spatial variogram model function.
    model_time : function
        Temporal variogram model function.
    model_metric : function
        Metric variogram model function.
    bins_space : numpy array of shape (n,)
        Space lag bins.
    bins_time : numpy array of shape (m,)
        Time lag bins.

    Returns
    -------
    Numpy array of shape (n, m)
        Predicted variogram values on grid.
    """
    variogram = np.empty(
        (len(bins_space), len(bins_time)), dtype=np.double,
    )
    for i, h in enumerate(bins_space):
        for j, t in enumerate(bins_time):
            variogram[i, j] = st_model(
                h, t, x, model_space, model_time, model_metric,
            )
    return variogram


def weighted_mean_square_error(
    x,
    st_model,
    model_space, model_time, model_metric,
    bins_space, bins_time,
    empirical_variogram,
    weights,
):
    """Calculate the weighted mean square error of the variogram model.
    Weighting is done by the number of samples in each bin divided by the
    squared distance.

    Parameters
    ----------
    x : tuple
        Variogram model parameters.
    st_model : function
        Spate-time variogram model function.
    model_space : function
        Spatial variogram model function.
    model_time : function
        Temporal variogram model function.
    model_metric : function
        Metric variogram model function.
    bins_space : numpy array of shape (n,)
        Space lag bins.
    bins_time : numpy array of shape (m,)
        Time lag bins.
    empirical_variogram : numpy array of shape (n, m)
        Empirical variogram, i.e. ground truth for the variogram model.
    weights : numpy array of shape (n, m)
        Number of samples in each bin.

    Returns
    -------
    float
        Weighted mean square error.
    """
    prediction = prediction_grid(
        x,
        st_model, model_space, model_time, model_metric,
        bins_space, bins_time,
    )
    error_grid = empirical_variogram - prediction
    return np.average(np.square(error_grid), weights=weights)


def calc_weights(bins_space, bins_time, ani, samples_per_bin):
    """Calculate the weights for fitting the variogram model.
    Weights are calculated as the number of samples in each bin divided by
    the squared distance (and normalized in the end).

    Parameters
    ----------
    bins_space : numpy array of shape (n,)
        Space lag bins.
    bins_time : numpy array of shape (m,)
        Time lag bins.
    ani : float
        Anisotropy factor.
    samples_per_bin : numpy array of shape (n, m)
        Number of samples in each bin.

    Returns
    -------
    Numpy array of shape (n, m)
        Weights for each bin.
    """

    n_space_bins = len(bins_space)
    n_time_bins = len(bins_time)

    distances_sq = np.square(ani * np.tile(bins_time, [n_space_bins, 1]))\
        + np.square(np.tile(np.expand_dims(bins_space, 0).T, [1, n_time_bins]))

    # maybe looks better with plain 1/distance?
    # but 1/distance_sq is in literature
    weights = samples_per_bin.astype(float) / distances_sq

    # normalize weights
    weights /= np.max(weights)

    return weights


def get_initial_parameters(
    model_str, empirical_variogram, space_dist_max, time_dist_max, ani,
):
    """Set reasonable initial parameters for the variogram model.

    Parameters
    ----------
    model_str : str
        Space-time variogram model to use ("sum", "product", "product_sum",
        "metric" or "sum_metric").
    empirical_variogram : numpy array of shape (n, m)
        Empirical variogram, i.e. ground truth for the variogram model.
    space_dist_max : float
        Maximum spatial distance in the empirical variogram.
    time_dist_max : float
        Maximum temporal distance in the empirical variogram.
    ani : float
        Anisotropy factor obtained by linear fitting.

    Returns
    -------
    Numpy array
        Initial parameters for the variogram model.
    """
    r_s = space_dist_max  # range space
    r_t = time_dist_max  # range time
    r_m = space_dist_max  # range metric

    c0_s = empirical_variogram[:, 0].max()  # sill space
    c0_t = empirical_variogram[0, :].max()  # sill time
    c0_m = empirical_variogram.max()  # sill metric

    b = empirical_variogram.min() / 2  # half nugget

    initial_params_dict = {
        "sum": [r_s, r_t, c0_s, c0_t, b, b],
        "product": [
            r_s, r_t, np.sqrt(c0_s), np.sqrt(c0_t), np.sqrt(b), np.sqrt(b),
        ],
        "product_sum": [
            r_s, r_t, np.sqrt(c0_s)/2, np.sqrt(c0_t)/2,
            np.sqrt(b/2), np.sqrt(b/2), 0.5,
        ],
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

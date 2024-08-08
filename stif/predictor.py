import pickle
import typing

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from scipy.optimize import minimize

try:
    from tensorflow import keras
    KERAS_INSTALLED = True
except ImportError:
    KERAS_INSTALLED = False

from stif import Data
from stif.utils import get_variogram
from stif.utils import get_covariogram
from stif.utils import calc_distance_matrix_1d
from stif.utils import calc_distance_matrix_2d
from stif.variogram_models import calc_weights
from stif.variogram_models import get_initial_parameters
from stif.variogram_models import variogram_model_dict
from stif.variogram_models import weighted_mean_square_error
plt.style.use("seaborn-v0_8-whitegrid")


if KERAS_INSTALLED:
    model_types = typing.Union[
        keras.models.Sequential,
        sklearn.base.BaseEstimator,
        None,
    ]
else:
    model_types = typing.Union[
        sklearn.base.BaseEstimator,
        None,
    ]


class Predictor:
    """Predictor class for space-time prediction.
    """

    def __init__(
        self,
        data: Data,
        covariate_model: model_types = None,
        cv_splits: int = 5,
        resampling=None,
    ):
        """Constructor for the Predictor class.

        Parameters
        ----------
        data : Data
            Data object containing the data.
        covariate_model : model_types, optional
            Covariate model to be used for prediction,
            can be either a Keras model (`keras.models.Sequential`)
            or a scikit-learn model (`sklearn.base.BaseEstimator`),
            by default None
        cv_splits : int, optional
            Number of cross validation splits for timeseries cross validation,
            by default 5
        resampling: imbalanced-learn sampler, optional
            Sampler for resampling the input before prediction, by default None
        """
        self._data = data
        self._cv_splits = cv_splits
        self._resampling = resampling

        self._cov_model = covariate_model
        self._X = self._data.get_training_covariates()
        self._y = self._data.predictand

        self._cross_val_res = None
        self._variogram = None
        self._variogram_bins_space = None
        self._variogram_bins_time = None
        self._variogram_model_function = None
        self._kriging_weights_function = None
        self._kriging_function = None

        self._is_binary = self._data.predictand.dtype == bool
        self._is_keras_model = self._cov_model.__class__.__module__ ==\
            "keras.src.engine.sequential"

        self._residuals = None

    def _prepare_geostatistics(self):
        self._residuals = self.get_residuals()

    def fit_covariate_model(self, train_idxs=slice(None)):
        """Fit the covariate model to the training data.

        Parameters
        ----------
        train_idxs : 1d numpy index, optional
            Indices of the samples used for training. Can be anything
            that is allowed for indexing a 1d numpy array, e.g. a slice,
            boolean array or tuple, by default all samples (`slice(None)`)
        """
        training_X = self._X[train_idxs, :]
        training_y = self._y[train_idxs]
        if self._resampling is not None:
            training_X, training_y = self._resampling.fit_resample(
                training_X, training_y,
            )

        if self._cov_model is not None:
            self._cov_model.fit(training_X, training_y)

    def save_covariate_model(self, filename):
        """Save the trained covariate model to a file.
        Used the Keras model saving function if its a Keras model,
        otherwise uses pickle.

        Parameters
        ----------
        filename : str
            Filename to save the model to.
        """
        if self._is_keras_model:
            self._cov_model.save(filename)
        else:
            with open(filename, 'wb') as file:
                pickle.dump(self._cov_model, file)

    def load_covariate_model(self, filename):
        """Load a trained covariate model from a file.
        Uses the Keras model loading function if its a Keras model,
        otherwise uses pickle.

        Parameters
        ----------
        filename : str
            Filename to load the model from.
        """
        if self._is_keras_model:
            if not KERAS_INSTALLED:
                raise ImportError(
                    "Keras is not installed."
                    "Please install Keras to load a Keras model.",
                )
            self._cov_model = keras.models.load_model(filename)
        else:
            with open(filename, 'rb') as file:
                self._cov_model = pickle.load(file)
        self._prepare_geostatistics()

    @property
    def _covariate_prediction_function(self):
        """Wrapper around the predict function of the covariate model.
        Difference between binary and non-binary models, as well as Keras
        and scikit-learn models are abstracted away.

        Returns
        -------
        function
            Function for prediction using the covariate model.
        """
        if self._is_binary and not self._is_keras_model:
            def res(X): return self._cov_model.predict_proba(X)[:, 1]
        else:
            res = self._cov_model.predict
        return res

    def get_covariate_prediction(self, idxs=slice(None)):
        """Covariate prediction on the input samples.
        Get the predicted covariate model prediction for the given indices.

        Parameters
        ----------
        idxs : 1d numpy index, optional
            Indices of the samples to predict. Can be anything
            that is allowed for indexing a 1d numpy array, e.g. a slice,
            boolean array or tuple, by default all samples (`slice(None)`)

        Returns
        -------
        1d numpy array
            Predicted covariate model predictions.
        """
        if self._cov_model is None:
            return np.zeros(len(idxs))
        else:
            return self._covariate_prediction_function(self._X[idxs]).flatten()

    def calc_covariate_prediction(self, df):
        """Covariate prediction on a pandas DataFrame.
        Get the predicted covariate model prediction for the given DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the data to predict on.

        Returns
        -------
        1d numpy array
            Predicted covariate model predictions.
        """
        if self._cov_model is None:
            return np.zeros(len(df))
        else:
            X = self._data.prepare_covariates(df)
            return self._covariate_prediction_function(X)

    def get_residuals(self, idxs=slice(None)):
        """Get the residuals of the covariate model.

        Parameters
        ----------
        idxs : 1d numpy index, optional
            Indices of the samples to predict. Can be anything
            that is allowed for indexing a 1d numpy array, e.g. a slice,
            boolean array or tuple, by default all samples (`slice(None)`)

        Returns
        -------
        1d numpy array
            Residuals of the covariate model.
        """
        if self._cov_model is not None:
            return self._y[idxs] - self.get_covariate_prediction(idxs)
        else:
            return self._y[idxs]

    def calc_cross_validation(
        self,
        kriging: bool = False,
        geostat_params: dict = dict(),
        max_test_samples: int = -1,
        verbose: bool = False,
        empirical_variogram_path: typing.Optional[str] = None,
    ):
        """Calculate cross validation results for the predictor.
        Can be used with the covariate model, kriging, or both.

        Parameters
        ----------
        kriging : bool, optional
            Use kriging (otherwise only covariate model), by default False
        geostat_params : dict, optional
            Parameters for empirical variogram, variogram model and kriging,
            by default empty dict.
        max_test_samples : int, optional
            Maximum number of test samples to consider per cross validation
            fold, by default `-1`, meaning no limit
        verbose : bool, optional
            Write status for each cross validation fold, by default False
        empirical_variogram_path : str or None, optional
            If set, empirical variogram will be read from file instead of
            calculated for every cross validation split (only relevant for
            `kriging==True`), by default None
        """
        cv = sklearn.model_selection.TimeSeriesSplit(n_splits=self._cv_splits)
        ground_truth_list = []
        prediction_list = []
        for fold, (train, test) in enumerate(cv.split(self._X, self._y)):
            if verbose:
                print("Fold {}".format(fold))
                print("\t train: {} samples".format(len(train)))
                print("\t test: {} samples".format(len(test)))
            if max_test_samples > 0:
                # randomly select max_test_samples from test set
                test = np.random.choice(
                    test,
                    size=min(max_test_samples, len(test)),
                    replace=False,
                )

            self.fit_covariate_model(train)
            ground_truth_list.append(self._y[test])
            prediction = self.get_covariate_prediction(test)
            if kriging:
                if "variogram_params" in geostat_params.keys():
                    variogram_params = geostat_params["variogram_params"]
                else:
                    variogram_params = dict()
                if "kriging_params" in geostat_params.keys():
                    kriging_params = geostat_params["kriging_params"]
                else:
                    kriging_params = dict()
                if "variogram_model_params" in geostat_params.keys():
                    variogram_model_params =\
                        geostat_params["variogram_model_params"]
                else:
                    variogram_model_params = dict()
                if empirical_variogram_path is None:
                    self.calc_empirical_variogram(train, **variogram_params)
                else:
                    self.load_empirical_variogram(empirical_variogram_path)
                self.fit_variogram_model(**variogram_model_params)
                kriging_mean, kriging_std = self.get_kriging_prediction(
                    self._data.space_coords[test, :],
                    self._data.time_coords[test],
                    leave_out_idxs=test,
                    **kriging_params,
                )
                prediction += kriging_mean

            prediction_list.append(prediction)
        self._cross_val_res = ground_truth_list, prediction_list

    def get_cross_val_metric(self, metric):
        """Calculate a metric for the cross validation results.
        Apply the gven metric function on each of the cross validation folds.

        Parameters
        ----------
        metric : function
            Metric function to be applied on the cross validation results,
            e.g. from sklearn.metrics.

        Returns
        -------
        List
            List of metric values for each cross validation fold.

        Raises
        ------
        ValueError
            Raises error if cross validation was not calculated before.
        """
        if self._cross_val_res is None:
            raise ValueError("Calc cross validation first.")

        ground_truth, prediction = self._cross_val_res

        res = []
        for i in range(self._cv_splits):
            res.append(metric(ground_truth[i], prediction[i]))
        return res

    def calc_empirical_variogram(
        self,
        idxs=slice(None),
        space_dist_max=3,
        time_dist_max=10,
        n_space_bins=10,
        n_time_bins=10,
        el_max=None,
    ):
        """Calculate the empirical space-time variogram for the given samples.
        Computation is done JIT-compiled using numba and without storing the
        entire distance matrix in memory.

        The basic formula for calculation is given by:

        .. math::
            \\gamma(h, u) = \\frac{1}{2 N(h, u)} \\sum_{i=1}^{N(h, u)} \\left[
            Z(\\mathbf{s}_i, t_i) - Z(\\mathbf{s}_i + h, t_i + u) \\right]^2

        Notes
        -----
        See [1]_, there called "sample variogram".

        Parameters
        ----------
        idxs : 1d numpy index, optional
            Specify the samples to include, by default all (`slice(None)`)
        space_dist_max : float, optional
            Maximum space lag, by default 3
        time_dist_max : float, optional
            Maximum time lag, by default 10
        n_space_bins : int, optional
            Number of spatial bins, by default 10
        n_time_bins : int, optional
            Number of temporal bins, by default 10
        el_max : int, optional
            Number of sample pairs to consider before termination, by default
            uses all sample pairs

        References
        ----------
        .. [1] G. B. M. Heuvelink, E. Pebesma, and B. Gräler, “Space-Time
               Geostatistics,” in Encyclopedia of GIS, S. Shekhar, H. Xiong,
               and X. Zhou, Eds., Cham: SpringerInternational Publishing, 2017,
               pp. 1919–1926. doi: 10.1007/978-3-319-17885-1_1647.

        """
        space_coords = self._data.space_coords[idxs, :]
        time_coords = self._data.time_coords[idxs]

        self._prepare_geostatistics()

        if self._residuals is None and self._cov_model is not None:
            raise ValueError(
                "Covariate model is set, but was not fitted yet. Please fit \
the covaraite model first, so the variogram can be calculated on the\
residuals")

        residuals = self._residuals[idxs]

        variogram, samples_per_bin, bin_width_space, bin_width_time =\
            get_variogram(
                space_coords,
                time_coords,
                residuals,
                space_dist_max,
                time_dist_max,
                n_space_bins,
                n_time_bins,
                el_max,
            )

        bins_space = np.arange(n_space_bins+1) * bin_width_space
        bins_space = ((bins_space[:-1] + bins_space[1:])/2)

        bins_time = np.arange(n_time_bins+1) * bin_width_time
        bins_time = ((bins_time[:-1] + bins_time[1:])/2)

        self._variogram = variogram
        self._variogram_bins_space = bins_space
        self._variogram_bins_time = bins_time
        self._variogram_samples_per_bin = samples_per_bin

    def calc_empirical_covariogram(
        self,
        idxs=slice(None),
        space_dist_max=3,
        time_dist_max=10,
        n_space_bins=10,
        n_time_bins=10,
        el_max=None,
    ):
        """Calculate the empirical space-time covariogram.
        Computation is done JIT-compiled using numba and without storing the
        entire distance matrix in memory.

        The basic formula for calculation is given by:

        .. math::
            \\rho(h, u) = \\frac{1}{\\sigma(Z) \\cdot N(h, u)}
            \\sum_{i=1}^{N(h, u)} \\left[
            (Z(\\mathbf{s}_i, t_i) - \\mu(Z)) \\cdot (Z(\\mathbf{s}_i + h,
            t_i + u) - \\mu(Z)) \\right]

        Parameters
        ----------
        idxs : 1d numpy index, optional
            Specify the samples to include, by default all (`slice(None)`)
        space_dist_max : float, optional
            Maximum space lag, by default 3
        time_dist_max : float, optional
            Maximum time lag, by default 10
        n_space_bins : int, optional
            Number of spatial bins, by default 10
        n_time_bins : int, optional
            Number of temporal bins, by default 10
        el_max : int, optional
            Number of sample pairs to consider before termination, by default
            uses all sample pairs
        """
        space_coords = self._data.space_coords[idxs, :]
        time_coords = self._data.time_coords[idxs]

        self._prepare_geostatistics()

        if self._residuals is None and self._cov_model is not None:
            raise ValueError(
                "Covariate model is set, but was not fitted yet. Please fit \
the covaraite model first, so the variogram can be calculated on the\
residuals")

        residuals = self._residuals[idxs]

        covariogram, samples_per_bin, bin_width_space, bin_width_time =\
            get_covariogram(
                space_coords,
                time_coords,
                residuals,
                space_dist_max,
                time_dist_max,
                n_space_bins,
                n_time_bins,
                el_max,
            )

        bins_space = np.arange(n_space_bins+1) * bin_width_space
        bins_space = ((bins_space[:-1] + bins_space[1:])/2)

        bins_time = np.arange(n_time_bins+1) * bin_width_time
        bins_time = ((bins_time[:-1] + bins_time[1:])/2)

        self._covariogram = covariogram
        self._covariogram_bins_space = bins_space
        self._covariogram_bins_time = bins_time
        self._covariogram_samples_per_bin = samples_per_bin

    def save_empirical_variogram(self, filename):
        """Save empirical variogram to file.
        Since the empirical variogram is expensive to compute, it can make
        sense to compute it once and store it, before fitting variogram models
        or Kriging.

        Parameters
        ----------
        filename : str
            Destination file name.

        Raises
        ------
        ValueError
            Variogram needs to be calculated before.
        """
        if self._variogram is None:
            raise ValueError("Calc empirical variogoram first.")
        np.savez(
            filename,
            variogram=self._variogram,
            bins_space=self._variogram_bins_space,
            bins_time=self._variogram_bins_time,
            samples_per_bin=self._variogram_samples_per_bin,
        )

    def load_empirical_variogram(self, filename):
        """Load empirical variogram from disc.

        Parameters
        ----------
        filename : str
            Filename to load from.
        """
        with np.load(filename) as data:
            self._variogram = data["variogram"]
            self._variogram_bins_space = data["bins_space"]
            self._variogram_bins_time = data["bins_time"]
            self._variogram_samples_per_bin = data["samples_per_bin"]

    def _create_variogram_model_function(self):
        if self._variogram_fit is None:
            raise ValueError("Fit variogram model first.")

        st_model, space_model, time_model, metric_model = \
            self._variogram_models

        st_model_fun = variogram_model_dict[st_model]
        space_model_fun = variogram_model_dict[space_model]
        time_model_fun = variogram_model_dict[time_model]
        metric_model_fun = variogram_model_dict[metric_model]
        x = self._variogram_fit.x

        @nb.vectorize([nb.float64(nb.float64, nb.float64)])
        def variogram_model(h, t):
            return st_model_fun(
                h,
                t,
                x,
                space_model_fun,
                time_model_fun,
                metric_model_fun,
            )

        return variogram_model

    def _create_kriging_weights_function(self):
        if self._variogram_model_function is None:
            raise ValueError("Create variogram model function first.")

        variogram_model_function = self._variogram_model_function

        @nb.njit(fastmath=True)
        def calc_kriging_weights(
            kriging_vector,
            coords_spatial,
            coords_temporal,
        ):
            n = len(kriging_vector)
            spatial_dist = calc_distance_matrix_2d(coords_spatial)
            temporal_dist = calc_distance_matrix_1d(coords_temporal)
            A_var = variogram_model_function(spatial_dist, temporal_dist)

            # for Lagrange multiplier
            A = np.ones((n+1, n+1), dtype=A_var.dtype)
            A[:-1, :-1] = A_var
            A[-1, -1] = 0

            b = kriging_vector

            # for Lagrange multiplier
            b = np.append(b, 1)

            w = np.linalg.lstsq(A, b)[0]

            # remove Lagrange multiplier
            w = w[:-1]

            return w
        return calc_kriging_weights

    def _create_kriging_function(self):
        if self._kriging_weights_function is None:
            raise ValueError("Create Kriging weights function first.")

        kriging_weights_function = self._kriging_weights_function
        variogram_model_function = self._variogram_model_function

        @nb.njit(fastmath=True, parallel=True)
        def nd_kriging(
            space, time,
            kriging_idxs,
            min_kriging_points, max_kriging_points,
            space_coords, time_coords,
        ):
            n_targets = len(time)
            kriging_weights = np.zeros(
                (n_targets, max_kriging_points), dtype=np.float32,
            )
            kriging_idx_matrix = np.zeros(
                (n_targets, max_kriging_points), dtype=np.uintc,
            )
            kriging_vectors = np.zeros(
                (n_targets, max_kriging_points), dtype=np.float32,
            )
            for target_i in nb.prange(n_targets):
                kriging_idxs_target = kriging_idxs[
                    kriging_idxs[:, 1]
                    == target_i, 0,
                ]
                if len(kriging_idxs_target) < min_kriging_points:
                    kriging_weights[target_i, :] = 0
                    continue
                h = np.sqrt(
                    np.sum(
                        np.square(
                            space_coords[kriging_idxs_target, :] -
                            space[target_i, :],
                        ), axis=1,
                    ),
                )
                t = np.abs(time_coords[kriging_idxs_target] - time[target_i])
                kriging_vector = variogram_model_function(h, t)
                if len(kriging_idxs_target) > max_kriging_points:
                    lowest_idxs = np.argsort(kriging_vector)[
                        :max_kriging_points
                    ]
                    kriging_idxs_target = kriging_idxs_target[lowest_idxs]
                    kriging_vector = kriging_vector[lowest_idxs]

                space_coords_local = space_coords[kriging_idxs_target, :]
                time_coords_local = time_coords[kriging_idxs_target]
                kriging_weights[
                    target_i,
                    :len(kriging_vector),
                ] = kriging_weights_function(
                    kriging_vector,
                    space_coords_local,
                    time_coords_local,
                )
                kriging_idx_matrix[target_i, :len(kriging_idxs_target)] =\
                    kriging_idxs_target

                kriging_vectors[target_i, :len(kriging_vector)] =\
                    kriging_vector
            return kriging_weights, kriging_vectors, kriging_idx_matrix
        return nd_kriging

    def fit_variogram_model(
        self,
        st_model="sum_metric",
        space_model="spherical",
        time_model="spherical",
        metric_model="spherical",
        plot_anisotropy=False,
    ):
        """Fit a variogram model to a precalculated empirical variogram.

        More details: [2]_

        Parameters
        ----------
        st_model : str, optional
            Model for combining the spatial and temporal component.
            Implemented are: "sum", "product", "product_sum", "metric" and
            "sum_metric", by default "sum_metric"
        space_model : str, optional
            Variogram model for the spatial component, can be "spherical"
            or "gaussian", by default "spherical"
        time_model : str, optional
            Variogram model for the temporal component, can be "spherical"
            or "gaussian", by default "spherical"
        metric_model : str, optional
            Variogram model for the metric component, can be "spherical"
            or "gaussian", by default "spherical"
        plot_anisotropy : bool, optional
            Shows a plot of the anisotropy fit if True, by default False

        References
        ----------
        .. [2] G. B. M. Heuvelink, E. Pebesma, and B. Gräler, “Space-Time
               Geostatistics,” in Encyclopedia of GIS, S. Shekhar, H. Xiong,
               and X. Zhou, Eds., Cham: SpringerInternational Publishing, 2017,
               pp. 1919–1926. doi: 10.1007/978-3-319-17885-1_1647.
        """
        slope_space = np.polynomial.polynomial.polyfit(
            self._variogram_bins_space, self._variogram[:, 0], deg=1,
        )[1]
        slope_time = np.polynomial.polynomial.polyfit(
            self._variogram_bins_time, self._variogram[0, :], deg=1,
        )[1]
        ani = slope_time / slope_space

        if plot_anisotropy:
            fig, ax = plt.subplots()
            ax.scatter(
                self._variogram_bins_space/ani,
                self._variogram[:, 0], label="rescaled spatial",
            )
            ax.scatter(
                self._variogram_bins_time,
                self._variogram[0, :], label="temporal",
            )
            ax.set_xlabel("lag")
            ax.set_ylabel("variance")
            ax.set_title("Anisotropy coefficient: {:.3}".format(ani))
            ax.legend()
            plt.show()

        weights = calc_weights(
            self._variogram_bins_space,
            self._variogram_bins_time,
            ani,
            self._variogram_samples_per_bin,
        )

        initial_params = get_initial_parameters(
            st_model,
            self._variogram,
            self._variogram_bins_space[-1],
            self._variogram_bins_time[-1],
            ani,
        )

        variogram_fit = minimize(
            weighted_mean_square_error,
            initial_params,
            args=(
                variogram_model_dict[st_model],
                variogram_model_dict[space_model],
                variogram_model_dict[time_model],
                variogram_model_dict[metric_model],
                self._variogram_bins_space,
                self._variogram_bins_time,
                self._variogram,
                weights,
            ),
            method="Nelder-Mead",
            options={"maxiter": 10000},
        )

        self._variogram_fit = variogram_fit
        self._variogram_models = [
            st_model, space_model, time_model, metric_model,
        ]
        self._variogram_model_function = \
            self._create_variogram_model_function()
        self._kriging_weights_function = \
            self._create_kriging_weights_function()
        self._kriging_function = \
            self._create_kriging_function()

    def _get_variogram_model_grid(self):
        if self._variogram_fit is None:
            raise ValueError("Fit variogram model first.")

        h, t = np.meshgrid(
            self._variogram_bins_space,
            self._variogram_bins_time,
            indexing="ij",
        )
        return self._variogram_model_function(h, t)

    def _get_kriging_weights(
        self,
        space,
        time,
        min_kriging_points=10,
        max_kriging_points=100,
        space_dist_max=None,
        time_dist_max=None,
        leave_out_idxs=None,
    ):
        if space_dist_max is None:
            space_dist_max = self._variogram_bins_space[-1]
        if time_dist_max is None:
            time_dist_max = self._variogram_bins_time[-1]

        kriging_idxs = self._data.get_kriging_idxs(
            space, time,
            space_dist_max,
            time_dist_max,
            leave_out_idxs,
        )

        return self._kriging_function(
            space, time,
            kriging_idxs,
            min_kriging_points, max_kriging_points,
            self._data.space_coords, self._data.time_coords,
        )

    def get_kriging_prediction(
        self,
        space,
        time,
        min_kriging_points=1,
        max_kriging_points=10,
        space_dist_max=None,
        time_dist_max=None,
        leave_out_idxs=None,
        batch_size=1000,
    ):
        """Get kriging prediction for the given space-time coordinates.
        The previously fitted variogram model is used together with the
        training data stored withing the `Predictor` object. The prediction is
        done in batches to avoid memory issues. Kriging weights are calculated
        using `numpy.lstsq`. Calculation is accelerated using numba JIT.

        Parameters
        ----------
        space : Numpy array of shape (n, 2).
            Spatial coordinates of the points to predict.
        time : Numpy array of shape (n,).
            Temporal coordinates of the points to predict.
        min_kriging_points : int, optional
            Minimum number of training points to consider, by default 1
        max_kriging_points : int, optional
            Maximum number of training points to consider, by default 10
        space_dist_max : int|None, optional
            Maximum spatial distance of training points to consider, by
            default None
        time_dist_max : int|None, optional
            Maximum temporal distance of training points to consider, by
            default None
        leave_out_idxs : numpy index, optional
            Indices of training points not to consider (e.g. for
            cross-validation), by default None
        batch_size : int, optional
            Number of samples to include in one batch (more samples is faster
            due to numba JIT compilation, but uses more memory), by default
            1000

        Returns
        -------
        tuple of two 1d numpy arrays
            Mean and standard deviation of the kriging prediction.
        """
        if self._kriging_function is None:
            raise ValueError(
                "No Kriging function defined. Did you fit a variogram model?")

        n_targets = len(time)
        if space_dist_max is None:
            space_dist_max = self._variogram_bins_space[-1] / 2
        if time_dist_max is None:
            time_dist_max = self._variogram_bins_time[-1] / 2

        kriging_mean = np.zeros(n_targets)
        kriging_std = np.zeros(n_targets)
        for i in range(0, n_targets, batch_size):
            batch_slice = slice(i, min(n_targets, i+batch_size))
            if leave_out_idxs is None:
                leave_out_idxs_batch = None
            else:
                leave_out_idxs_batch = leave_out_idxs[batch_slice]
            kriging_mean[batch_slice], kriging_std[batch_slice] = \
                self._get_kriging_prediction_batch(
                    space[batch_slice, :],
                    time[batch_slice],
                    min_kriging_points,
                    max_kriging_points,
                    space_dist_max,
                    time_dist_max,
                    leave_out_idxs_batch,
            )
        return kriging_mean, kriging_std

    def _get_kriging_prediction_batch(
        self,
        space,
        time,
        min_kriging_points,
        max_kriging_points,
        space_dist_max,
        time_dist_max,
        leave_out_idxs,
    ):
        w, kriging_vectors, kriging_idx_matrix = self._get_kriging_weights(
            space, time,
            min_kriging_points,
            max_kriging_points,
            space_dist_max,
            time_dist_max,
            leave_out_idxs,
        )
        kriging_mean = np.sum(
            w * self._residuals[kriging_idx_matrix],
            axis=1,
        )

        kriging_std = np.sqrt(np.sum(w * kriging_vectors, axis=1))
        return kriging_mean, kriging_std

    def predict(self, df, kriging_params=dict()):
        covariate_prediction = self.calc_covariate_prediction(df)
        space = df[self._data._space_cols].to_numpy()
        time = df[self._data._time_col].to_numpy()
        kriging_mean, kriging_std = self.get_kriging_prediction(
            space,
            time,
            **kriging_params
        )
        return covariate_prediction + kriging_mean, kriging_std

    def plot_kriging_weights(
        self,
        space,
        time,
        min_kriging_points=10,
        max_kriging_points=100,
        target="screen",
    ):
        """Plot the kriging weights for a given space-time coordinate.

        Parameters
        ----------
        space : iterable of length 2
            Space coordinates of the point to plot.
        time : float
            Time coordinate of the point to plot.
        min_kriging_points : int, optional
            Minimum number of Kriging points, by default 10
        max_kriging_points : int, optional
            Maximum number of Kriging points, by default 100
        target : str, optional
            If not "screen", path to write the figure to, by default "screen"
        """
        w, kriging_vectors, kriging_idx_matrix = \
            self._get_kriging_weights(
                np.array([space]), np.array([time]),
                min_kriging_points,
                max_kriging_points,
            )
        w = w[0, :]
        kriging_idxs = kriging_idx_matrix[0, :]

        space_coords = self._data.space_coords[kriging_idxs, :]
        time_coords = self._data.time_coords[kriging_idxs]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        plot = ax.scatter(
            space_coords[:, 0],
            space_coords[:, 1], time_coords, s=w*100,
            c=self._residuals[kriging_idxs],
            cmap="viridis",
        )
        ax.scatter([space[0]], [space[1]], [time], color="xkcd:red")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Time")
        fig.colorbar(plot)
        if target == "screen":
            plt.show()
        else:
            fig.savefig(target)

    def plot_cross_validation_roc(self, target="screen"):
        """Plot ROC curve for the cross validation results.

        Parameters
        ----------
        target : str, optional
            If not "screen", path to write the figure to, by default "screen"

        Raises
        ------
        ValueError
            Raises error if cross validation was not calculated before.
        """
        if self._cross_val_res is None:
            raise ValueError("Calc cross validation first.")

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(figsize=(6, 6))

        ground_truth, pred = self._cross_val_res

        for fold in range(self._cv_splits):
            viz = sklearn.metrics.RocCurveDisplay.from_predictions(
                ground_truth[fold],
                pred[fold],
                name=f"ROC fold {fold}",
                alpha=0.3,
                lw=1,
                ax=ax,
                # plot_manually for compatibility with older sklearn and cuml
                # plot_chance_level=(fold == self._cv_splits - 1),
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], color="k")

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (
                mean_auc, std_auc,
            ),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Mean ROC curve with variability\n(TimeSeriesPrediction)",
        )
        ax.axis("square")
        ax.legend(loc="lower right")

        if target == "screen":
            plt.show()
        else:
            fig.savefig(target)

    def plot_empirical_variogram(
        self,
        fig=None,
        ax=None,
        vrange=(None, None),
        title="",
        target="screen",
    ):
        """Plot the empirical variogram in a surface plot.
        Semivariance over space and time lags.

        Parameters
        ----------
        fig : matplotlib figure, optional
            Existing figure to plot the variogram to (needs also `ax` to be
            set), by default None
        ax : matplotlib axis, optional
            Existing axis to plot the variogram to, by default None
        vrange : tuple(float), optional
            Value range minimum and maximum, by default (None, None)
        title : str, optional
            Axis title, by default ""
        target : str, optional
            If not "screen", path to write the figure to, by default "screen"

        Raises
        ------
        ValueError
            _description_
        """
        if self._variogram is None:
            raise ValueError("Calc variogram first.")

        self._plot_variogram(
            self._variogram,
            fig=fig,
            ax=ax,
            vrange=vrange,
            title=title,
            target=target,
        )

    def plot_empirical_covariogram(
        self,
        fig=None,
        ax=None,
        vrange=(None, None),
        title="",
        target="screen",
    ):
        """Plot the empirical covariogram in a surface plot.
        Autocorrelation over space and time lags.

        Parameters
        ----------
        fig : matplotlib figure, optional
            Existing figure to plot the covariogram to (needs also `ax` to be
            set), by default None
        ax : matplotlib axis, optional
            Existing axis to plot the covariogram to, by default None
        vrange : tuple(float), optional
            Value range minimum and maximum, by default (None, None)
        title : str, optional
            Axis title, by default ""
        target : str, optional
            If not "screen", path to write the figure to, by default "screen"

        Raises
        ------
        ValueError
            _description_
        """
        if self._covariogram is None:
            raise ValueError("Calc variogram first.")

        self._plot_covariogram(
            self._covariogram,
            fig=fig,
            ax=ax,
            vrange=vrange,
            title=title,
            target=target,
        )

    def _plot_variogram(
        self,
        variogram,
        fig=None,
        ax=None,
        vrange=(None, None),
        title="",
        target="screen",
    ):
        if self._variogram_bins_space is None:
            raise ValueError("Calc variogram first or set bins manually.")

        X, Y = np.meshgrid(
            self._variogram_bins_space,
            self._variogram_bins_time,
        )
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = plt.axes(projection='3d')
        vmin, vmax = vrange
        plot = ax.plot_surface(
            X, Y, variogram.T, vmin=vmin, vmax=vmax,
            cmap="plasma", edgecolor="black", linewidth=0.5,
        )
        ax.view_init(elev=35., azim=225)
        ax.grid(False)
        ax.set_title(title)
        ax.set_xlabel("space lag")
        ax.set_ylabel("time lag")
        fig.colorbar(plot, fraction=0.044, pad=0.04)

        if ax is None:
            if target == "screen":
                plt.show()
            else:
                fig.savefig(target)

    def _plot_covariogram(
        self,
        covariogram,
        fig=None,
        ax=None,
        vrange=(None, None),
        title="",
        target="screen",
    ):
        if self._covariogram_bins_space is None:
            raise ValueError("Calc variogram first or set bins manually.")

        X, Y = np.meshgrid(
            self._covariogram_bins_space,
            self._covariogram_bins_time,
        )
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = plt.axes(projection='3d')
        vmin, vmax = vrange
        plot = ax.plot_surface(
            X, Y, covariogram.T, vmin=vmin, vmax=vmax,
            cmap="plasma", edgecolor="black", linewidth=0.5,
        )
        ax.view_init(elev=35., azim=225)
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.grid(False)
        ax.set_title(title)
        ax.set_xlabel("space lag")
        ax.set_ylabel("time lag")
        ax.set_zlabel('covariance')
        fig.colorbar(plot, fraction=0.044, pad=0.04)

        if ax is None:
            if target == "screen":
                plt.show()
            else:
                fig.savefig(target)

    def plot_variogram_model_comparison(
        self,
        space_model="spherical",
        time_model="spherical",
        metric_model="spherical",
        target="screen",
    ):
        """Plot the empirical variogram and the fitted variogram models.

        Parameters
        ----------
        space_model : str, optional
            Model for spatial variogram component ("spherical" or "gaussian"),
            by default "spherical"
        time_model : str, optional
            Model for temporal variogram component ("spherical" or "gaussian"),
            by default "spherical"
        metric_model : str, optional
            Model for metric variogram component ("spherical" or "gaussian"),
            by default "spherical"
        target : str, optional
            If not "screen", path to write the figure to, by default "screen"
        """
        fig = plt.figure(figsize=(17, 10))
        ax = fig.add_subplot(2, 3, 1, projection='3d')
        self.plot_empirical_variogram(fig, ax, title="empirical")

        models = ["sum", "product", "product_sum", "metric", "sum_metric"]
        for i, model in enumerate(models):
            self.fit_variogram_model(
                st_model=model,
                space_model=space_model,
                time_model=time_model,
                metric_model=metric_model,
            )
            grid = self._get_variogram_model_grid()
            wmse = self._variogram_fit.fun
            ax = fig.add_subplot(2, 3, i+2, projection='3d')
            self._plot_variogram(
                grid, fig, ax,
                title="{} (WMSE={:.6f})".format(model, wmse),
            )
        fig.tight_layout()
        if target == "screen":
            plt.show()
        else:
            fig.savefig(target)

    def plot_cross_validation_residuals(self, target="screen"):
        """Plot residuals for the cross validation results.

        Parameters
        ----------
        target : str, optional
            If not "screen", path to write the figure to, by default "screen"

        Raises
        ------
        ValueError
            Raises error if cross validation was not calculated before.
        """
        if self._cross_val_res is None:
            raise ValueError("Calc cross validation first.")

        fig, ax = plt.subplots(figsize=(6, 6))

        ground_truth, pred = self._cross_val_res

        maximum = 0
        minimum = np.inf
        for fold in range(self._cv_splits):
            maximum = max(
                maximum,
                np.max(pred[fold]),
                np.max(ground_truth[fold]),
            )
            minimum = min(
                minimum,
                np.min(pred[fold]),
                np.min(ground_truth[fold]),
            )
            r_squared = sklearn.metrics.r2_score(
                ground_truth[fold], pred[fold],
            )
            ax.scatter(
                ground_truth[fold],
                pred[fold],
                label=f"Fold {fold}: r2={r_squared:.3f}",
            )

        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            color="k",
        )

        ax.set(
            xlabel="Ground truth",
            ylabel="Prediction",
            title="Cross validation\n(TimeSeriesPrediction)",
        )
        ax.axis("square")
        ax.legend(loc="lower right")

        if target == "screen":
            plt.show()
        else:
            fig.savefig(target)

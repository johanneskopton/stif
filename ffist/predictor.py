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

from ffist import Data
from ffist.utils import get_distances
from ffist.utils import histogram2d
from ffist.utils import calc_distance_matrix_1d
from ffist.utils import calc_distance_matrix_2d
from ffist.variogram_models import calc_weights
from ffist.variogram_models import get_initial_parameters
from ffist.variogram_models import variogram_model_dict
from ffist.variogram_models import weighted_mean_square_error
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
    def __init__(
        self,
        data: Data,
        covariate_model: model_types = None,
        cv_splits: int = 5,
    ):
        self._data = data
        self._cv_splits = cv_splits

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
        self._cov_model.fit(self._X[train_idxs, :], self._y[train_idxs])
        self._prepare_geostatistics()

    def save_covariate_model(self, filename):
        if self._is_keras_model:
            self._cov_model.save(filename)
        else:
            with open(filename, 'wb') as file:
                pickle.dump(self._cov_model, file)

    def load_covariate_model(self, filename):
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
        if self._is_binary and not self._is_keras_model:
            def res(X): return self._cov_model.predict_proba(X)[:, 1]
        else:
            res = self._cov_model.predict
        return res

    def get_covariate_probability(self, idxs=slice(None)):
        return self._covariate_prediction_function(self._X[idxs])

    def predict_covariate_probability(self, df):
        X = self._data.prepare_covariates(df)
        return self._cov_model.predict(X)

    def get_residuals(self, idxs=slice(None)):
        return self.get_covariate_probability(idxs) - self._y[idxs]

    def calc_cross_validation(self):
        cv = sklearn.model_selection.TimeSeriesSplit(n_splits=self._cv_splits)
        ground_truth = []
        prediction = []
        for fold, (train, test) in enumerate(cv.split(self._X, self._y)):
            self.fit_covariate_model(train)
            ground_truth.append(self._y[test])
            prediction.append(self.get_covariate_probability(test))

        self._cross_val_res = ground_truth, prediction

    def get_cross_val_metric(self, metric):
        if self._cross_val_res is None:
            self.calc_cross_validation()

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
        el_max=1e6,
    ):
        space_coords = self._data.space_coords[idxs, :]
        time_coords = self._data.time_coords[idxs]
        residuals = self._residuals[idxs]

        space_lags, time_lags, sq_val_delta = get_distances(
            space_coords,
            time_coords,
            residuals,
            space_dist_max,
            time_dist_max,
            el_max,
        )

        space_range = (0, space_dist_max)
        time_range = (0, time_dist_max)
        hist, samples_per_bin, bin_width_space, bin_width_time = histogram2d(
            space_lags,
            time_lags,
            n_space_bins,
            n_time_bins,
            space_range,
            time_range,
            sq_val_delta,
        )

        # I think this "/2" is necessary, because in samples_per_bin are only
        # n^2/2 samples in total
        variogram = np.divide(
            hist,
            samples_per_bin,
            out=np.ones_like(hist) * np.nan,
            where=samples_per_bin != 0,
        ) / 2

        bins_space = np.arange(n_space_bins+1) * bin_width_space
        bins_space = ((bins_space[:-1] + bins_space[1:])/2)

        bins_time = np.arange(n_time_bins+1) * bin_width_time
        bins_time = ((bins_time[:-1] + bins_time[1:])/2)

        self._variogram = variogram
        self._variogram_bins_space = bins_space
        self._variogram_bins_time = bins_time
        self._variogram_samples_per_bin = samples_per_bin

    def save_empirical_variogram(self, filename):
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
            for target_i in nb.prange(n_targets):
                kriging_idxs_target = kriging_idxs[
                    kriging_idxs[:, 1]
                    == target_i, 0,
                ]
                if len(kriging_idxs_target) < min_kriging_points:
                    kriging_weights[target_i, :] = np.nan
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
            return kriging_weights, kriging_idx_matrix
        return nd_kriging

    def fit_variogram_model(
        self,
        st_model="sum_metric",
        space_model="spherical",
        time_model="spherical",
        metric_model="spherical",
        plot_anisotropy=False,
    ):
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

    def get_variogram_model_grid(self):
        if self._variogram_fit is None:
            raise ValueError("Fit variogram model first.")

        h, t = np.meshgrid(
            self._variogram_bins_space,
            self._variogram_bins_time,
            indexing="ij",
        )
        return self._variogram_model_function(h, t)

    def get_kriging_weights(
        self,
        space,
        time,
        min_kriging_points=10,
        max_kriging_points=100,
        space_dist_max=None,
        time_dist_max=None,
    ):
        if space_dist_max is None:
            space_dist_max = self._variogram_bins_space[-1]
        if time_dist_max is None:
            time_dist_max = self._variogram_bins_time[-1]
        kriging_idxs = self._data.get_kriging_idxs(
            space, time,
            space_dist_max,
            time_dist_max,
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
        min_kriging_points=10,
        max_kriging_points=100,
        space_dist_max=None,
        time_dist_max=None,
    ):
        w, kriging_idx_matrix = self.get_kriging_weights(
            space, time,
            min_kriging_points,
            max_kriging_points,
            space_dist_max,
            time_dist_max,
        )
        return np.sum(w * self._residuals[kriging_idx_matrix], axis=1)

    def plot_kriging_weights(
        self,
        space,
        time,
        min_kriging_points=10,
        max_kriging_points=100,
        target="screen",
    ):
        w, kriging_idx_matrix = self.get_kriging_weights(
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
        if self._cross_val_res is None:
            self.calc_cross_validation()

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
    ):
        if self._variogram is None:
            raise ValueError("Calc variogram first.")

        self._plot_variogram(
            self._variogram,
            fig=fig,
            ax=ax,
            vrange=vrange,
            title=title,
        )

    def _plot_variogram(
        self,
        variogram,
        fig=None,
        ax=None,
        vrange=(None, None),
        title="",
    ):
        if self._variogram_bins_space is None:
            raise ValueError("Calc variogram first or set bins manually.")

        X, Y = np.meshgrid(
            self._variogram_bins_space,
            self._variogram_bins_time,
        )
        if ax is None:
            standalone = True
            fig = plt.figure(figsize=(5, 5))
            ax = plt.axes(projection='3d')
        else:
            standalone = False
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

        if standalone:
            plt.show()

    def plot_variogram_model_comparison(
        self,
        space_model="spherical",
        time_model="spherical",
        metric_model="spherical",
        target="screen",
    ):
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
            grid = self.get_variogram_model_grid()
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

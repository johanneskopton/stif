import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from scipy.optimize import minimize

from ffist import Data
from ffist.utils import get_distances
from ffist.utils import histogram2d
from ffist.variogram_models import calc_weights
from ffist.variogram_models import get_initial_parameters
from ffist.variogram_models import variogram_model_dict
from ffist.variogram_models import weighted_mean_square_error
plt.style.use("seaborn-v0_8-whitegrid")


class Predictor:
    def __init__(
        self,
        data: Data,
        covariate_model: sklearn.base.ClassifierMixin,
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

        self._is_binary = self._data.predictand.dtype == bool

    def fit_covariate_model(self, train_idxs=None):
        if train_idxs is None:
            self._cov_model.fit(self._X, self._y)
        else:
            self._cov_model.fit(self._X[train_idxs, :], self._y[train_idxs])

    @property
    def _covariate_prediction_function(self):
        if self._is_binary and \
                self._cov_model.__class__.__module__ !=\
                "keras.src.engine.sequential":
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
        idxs=slice(None, None),
        space_dist_max=3,
        time_dist_max=10,
        n_space_bins=10,
        n_time_bins=10,
        el_max=1e6,
    ):
        residuals = self.get_residuals(idxs)
        space_coords = self._data.space_coords[idxs, :]
        time_coords = self._data.time_coords[idxs]

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

    def get_variogram_model_grid(self):
        if self._variogram_fit is None:
            raise ValueError("Fit variogram model first.")

        h, t = np.meshgrid(
            self._variogram_bins_space,
            self._variogram_bins_time,
            indexing="ij",
        )
        return self._variogram_model_function(h, t)

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
                title="{} (WMSE={:.3f})".format(model, wmse),
            )
        fig.tight_layout()
        if target == "screen":
            plt.show()
        else:
            fig.savefig(target)

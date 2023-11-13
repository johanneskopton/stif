from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import sklearn.model_selection

from occurence.occurence_data import OccurenceData
plt.style.use("ggplot")


class CovariancePredictor:
    def __init__(
        self,
        data: OccurenceData,
        model: sklearn.base.ClassifierMixin,
        covariate_cols,
        covariate_transformations=dict(),
        test_size=0.1,
        test_split_forecast=True,
        model_params=dict(),
    ):
        self._data = data
        self._model = model(**model_params)

        self._X = data.get_covariates(
            covariate_cols, covariate_transformations,
        )
        self._y = data.presence

        if test_split_forecast:
            n_test_samples = int(len(self._data.time_coords) * test_size)
            test_idxs = np.argsort(self._data.time_coords)[-n_test_samples:]
            self._X_train = np.delete(self._X, test_idxs, axis=0)
            self._X_test = self._X[test_idxs, :]
            self._y_train = np.delete(self._y, test_idxs)
            self._y_test = self._y[test_idxs]
        else:
            self._X_train, self._X_test, self._y_train, self._y_test = \
                sklearn.model_selection.train_test_split(
                    self._X, self._y, test_size=test_size, random_state=0,
                )

    def fit(self):
        self._model.fit(self._X_train, self._y_train)

    @cached_property
    def test_prediction(self):
        return self._model.predict(self._X_test)

    @cached_property
    def test_probability(self):
        return self._model.predict_proba(self._X_test)[:, 1]

    @property
    def confusion_matrix(self):
        return sklearn.metrics.confusion_matrix(
            self._y_test,
            self.test_prediction,
        )

    @property
    def roc_auc_score(self):
        return sklearn.metrics.roc_auc_score(
            self._y_test, self.test_probability,
        )

    def plot_roc_curve(self):
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            self._y_test, self.test_probability,
        )
        auc = self.roc_auc_score
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label="data 1, auc="+str(auc))
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend(loc=4)
        plt.show()

import math
from functools import cached_property

import numba as nb
import numpy as np
import pandas as pd


@nb.njit(fastmath=True)
def sinusodial_feature_transform(x, n_freqs=6, full_circle=None):
    if full_circle is None:
        full_circle = x.max() - x.min()

    x = x.copy()
    x = x / full_circle * 2 * math.pi
    res = np.empty((len(x), n_freqs), dtype=float)

    for i in range(n_freqs):
        if i % 2 == 0:
            res[:, i] = np.sin(x * (1 + int(i/2)))
        else:
            res[:, i] = np.cos(x * (1 + int(i/2)))
    return res


class Data:
    def __init__(
        self,
        df: pd.DataFrame,
        space_cols=["longitude", "latitude"],
        time_col=None,
        predictand_col="predictand",
        covariate_cols=list(),
        normalize=True,
        covariate_transformations=dict(),
    ):
        """Prepare and provide the data.

        Parameters
        ----------
        df : pandas.DataFrame
            The unstructured space-time data with one observation per row.
        space_cols : list(str), optional
            The names of the spatial coordinate columns,
            by default ["longitude", "latitude"]
        time_col : str, optional
            The name of the temporal coordinate column, by default "time"
        predictand_col : str, optional
            Name of the column, that contains the predictand information,
            by default "predictand"
        covariate_cols : list(str), optional
            The names of the covariate columns, by default empty list.
        normalize : bool, optional
            Provide normalized coordinates to the predictors, by default True
        covariate_transformations : dict, optional
            Functions for transforming the respective columns
        """
        self._training_df = df
        self._space_cols = space_cols
        self._time_col = time_col
        self._predictand_col = predictand_col
        self._covariate_cols = covariate_cols
        self._normalize = normalize
        self._covariate_transformations = covariate_transformations

    @cached_property
    def normalization_bounds(self):
        normalization_bounds = dict()
        for column in self._training_df.columns:
            if self._training_df[column].dtype.kind in 'iuf':
                col_min = self._training_df[column].min()
                col_max = self._training_df[column].max()
                normalization_bounds[column] = col_min, col_max
        return normalization_bounds

    def normalize(self, array, col_name):
        norm_min, norm_max = self.normalization_bounds[col_name]
        return (array - norm_min) / (norm_max - norm_min)

    @property
    def space_coords(self):
        return self._training_df[self._space_cols].to_numpy()

    @property
    def time_coords(self):
        return self._training_df[self._time_col].to_numpy()

    @property
    def predictand(self):
        return self._training_df[self._predictand_col].to_numpy()

    def prepare_covariates(self, df):
        X = np.empty((len(df), 0), dtype=float)
        for i, covariate_col in enumerate(self._covariate_cols):
            col_array = df[covariate_col].to_numpy(copy=True)
            if covariate_col in self._covariate_transformations.keys():
                col_trans = self._covariate_transformations[covariate_col](
                    col_array,
                )
                X = np.c_[X, col_trans]
            if self._normalize:
                col_array = self.normalize(col_array, covariate_col)
            X = np.c_[X, col_array]
        return X

    def get_training_covariates(self):
        return self.prepare_covariates(self._training_df)

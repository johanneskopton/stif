import math
from functools import cached_property

import numba as nb
import numpy as np
import pandas as pd
import scipy.spatial.distance


@nb.njit(fastmath=True)
def sinusoidal_feature_transform(x, n_freqs=6, full_circle=None):
    """Transform a 1D array into a sinusoidal feature space.
    Helpful for transforming the time dimension to better capture
    periodic (e.g. seasonal) patterns.

    Parameters
    ----------
    x : numpy array of shape (n,)
        The input array to be transformed
    n_freqs : int, optional
        Number of frequencies, by default 6
    full_circle : float|None, optional
        Period length, e.g. one year, by default `x.max()-x.min()`

    Returns
    -------
    numpy array of shape (n, n_freqs)
        The transformed array
    """
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
    def _normalization_bounds(self):
        normalization_bounds = dict()
        for column in self._training_df.columns:
            if self._training_df[column].dtype.kind in 'iuf':
                col_min = self._training_df[column].min()
                col_max = self._training_df[column].max()
                normalization_bounds[column] = col_min, col_max
        return normalization_bounds

    def normalize(self, array, col_name):
        norm_min, norm_max = self._normalization_bounds[col_name]
        return (array - norm_min) / (norm_max - norm_min)

    @property
    def space_coords(self):
        """Get the spatial coordinates of the training data.

        Returns
        -------
        Numpy array of shape (n, 2)
            The spatial coordinates
        """
        return self._training_df[self._space_cols].to_numpy()

    @property
    def time_coords(self):
        """Get the temporal coordinates of the training data.

        Returns
        -------
        Numpy array of shape (n,)
            The temporal coordinates
        """
        return self._training_df[self._time_col].to_numpy()

    @property
    def predictand(self):
        """Get the predictand values of the training data.

        Returns
        -------
        Numpy array of shape (n,)
            The predictand values
        """
        return self._training_df[self._predictand_col].to_numpy()

    def prepare_covariates(self, df):
        """Transform the covariates from a Dataframe with train or test data.
        To apply the prediction model on unseen data, the covariates
        need to be transformed in the same way as the training data.

        Parameters
        ----------
        df : pandas.DataFrame
            The unstructured space-time data with one observation per row.

        Returns
        -------
        numpy array of shape (n, m)
            The transformed covariates
        """
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
        """Get the transformed covariates of the training data.

        Returns
        -------
        Numpy array of shape (n, m)
            The transformed covariates
        """
        return self.prepare_covariates(self._training_df)

    def get_kriging_idxs(
            self,
            space,
            time,
            space_dist_max,
            time_dist_max,
            leave_out_idxs,
    ):
        """Get the indices of the training data that are within the
        specified spatial and temporal distance.

        Parameters
        ----------
        space : numpy array of shape (n, 2)
            The spatial coordinates
        time : numpy array of shape (n,)
            The temporal coordinates
        space_dist_max : float
            Maximum spatial distance
        time_dist_max : float
            Maximum temporal distance
        leave_out_idxs : numpy index
            The indices to be left out (e.g. for cross-validation)
        """
        # space_dist_sq = np.sum((self.space_coords - space)**2, axis=1)
        space_dist_sq = scipy.spatial.distance.cdist(
            self.space_coords,
            space,
            metric='sqeuclidean',
        )
        time_dist = scipy.spatial.distance.cdist(
            self.time_coords.reshape(-1, 1), time.reshape(-1, 1),
        )
        index_mask = np.ones([len(self.time_coords), len(time)], dtype=bool)
        if leave_out_idxs is not None:
            index_mask[leave_out_idxs, np.arange(len(time))] = False
        is_close_enough = (
            (space_dist_sq < space_dist_max**2) &
            (time_dist < time_dist_max) &
            (self.time_coords.reshape(-1, 1) <= time.reshape(1, -1)) &
            index_mask
        )
        return np.column_stack(np.nonzero(is_close_enough))

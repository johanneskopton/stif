class OccurenceData:
    def __init__(
        self,
        df,
        normalize=True,
        space_cols=["longitude", "latitude"],
        time_col="time",
        presence_col="presence",
    ):
        """Prepare and provide the data.

        Parameters
        ----------
        df : pandas.DataFrame
            The unstructured space-time data with one observation per row.
        normalize : bool, optional
            Provide normalized coordinates to the predictors, by default True
        space_cols : list(str), optional
            The names of the spatial coordinate columns,
            by default ["longitude", "latitude"]
        time_col : str, optional
            The name of the temporal coordinate column, by default "time"
        presence_col : str, optional
            Name of the column, that contains the presence information
            with presence:=True and absence:=False, by default "presence"
        """
        self._df = df
        self._space_cols = space_cols
        self._time_col = time_col
        self._presence_col = presence_col

        if normalize:
            self._space_min = self._df[space_cols].min().to_numpy()
            self._space_max = self._df[space_cols].max().to_numpy()
            self._time_min = self._df[time_col].min()
            self._time_max = self._df[time_col].max()
            self._df[space_cols] = (self._df[space_cols] - self._space_min) / \
                (self._space_max - self._space_min)
            self._df[time_col] = (self._df[time_col] - self._time_min) / \
                (self._time_max - self._time_min)

    @property
    def space_coords(self):
        return self._df[self._space_cols].to_numpy()

    @property
    def time_coords(self):
        return self._df[self._time_col].to_numpy()

    @property
    def presence(self):
        return self._df[self._presence_col].to_numpy()

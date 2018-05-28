from sklearn.base import BaseEstimator, TransformerMixin

__numeric_f_time__ = ["%d", "%H", "%I", "%j", "%m", "%M", "%S", "%U", "%w", "%W", "%y"]


class AddDateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Adds calendar features (hour, weekday, day of year, year, etc...)

    Parameters
    ----------
    :param feat: a dictionary with the date and/or time features to be extracted with key as the desired column name.
    Visit http://strftime.org/ for Python's strftime directives which can used here.
    :param index: bool or str (default=True)
    if default=True The index will be used for extraction of the date and/or time features
    else a column name containing datetime should be specified

    Example:
            dataset = generate_ts(n_samples=1000, n_features=1, n_targets=1,
                                    split_X_y=False, start_date='2016-01-03 00:00',
                                    freq='1H')
            c = AddDateTimeFeatures({'hour': '%H'})

    """

    def __init__(self, feat, index=True):
        self.feat = feat
        self._index = index

    def fit(self, X, y=None):
        return self  # nothing else do do

    def transform(self, X, y=None):
        if self._index is True:
            return self._index_transform(X, y)
        elif self._index in X.columns:
            return self._column_transform(X, y)

    def _index_transform(self, X, y=None):
        if isinstance(self.feat, dict):
            for key, value in self.feat.items():
                X[key] = X.index.strftime(value)
                if value in __numeric_f_time__:
                    X[key] = X[key].astype(int)
            return X
        else:
            raise ValueError('feat parameter should be a dictionary!')

    def _column_transform(self, X, y=None):
        if isinstance(self.feat, dict):
            for key, value in self.feat.items():
                X[key] = X[self._index].dt.strftime(value)
                if value in __numeric_f_time__:
                    X[key] = X[key].astype(int)
            return X
        else:
            raise ValueError('feat parameter should be a dictionary!')

    def inverse_transform(self, X, y=None):
        return X

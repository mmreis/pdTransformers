from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class AddCircular(BaseEstimator, TransformerMixin):
    """
    Transforms circular features into its co-sen and sen parts

    Parameters
    ----------
    :param dict_var: a dictionary containing the name of the column as key and the maximum period as value.
    :param drop_original: bool (default=True) if True the original column is drooped from the returned data set

    Example:
            dataset = generate_ts(n_samples=1000, n_features=1, n_targets=1,
                                split_X_y=False, start_date='2016-01-03 00:00',
                                freq='1H')
            c = AddCircular({'hour': 24, 'wday': 7, 'month': 12})

    """

    def __init__(self, dict_var=None, drop_original=True):
        self.dict_var = dict_var
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self  # nothing else do do

    def transform(self, X, y=None):
        for var, period in self.dict_var.items():
            # check if variable exists
            if var not in X.columns:
                print(
                    ' ## ERROR! Sine and Cosine columns from \'{}\' were excluded, '
                    'since the it is not present in data set'.format(
                        var))
                continue

            X['%s_c' % var] = np.cos(2 * np.pi * X[var] / period)
            X['%s_s' % var] = np.sin(2 * np.pi * X[var] / period)
            if self.drop_original:
                X.drop(var, axis=1, inplace=True)
        return X

    def inverse_transform(self, X, y=None):
        return X

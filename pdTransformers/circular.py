from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class AddCircular(BaseEstimator, TransformerMixin):
    def __init__(self, dict_var=None, drop_original=True):
        self.dict_var = dict_var
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self  # nothing else do do

    def transform(self, X, y=None):
        for var, period in self.dict_var.items():
            X['%s_c' % var] = np.cos(2 * np.pi * X[var] / period)
            X['%s_s' % var] = np.sin(2 * np.pi * X[var] / period)
            if self.drop_original:
                X.drop(var, axis=1, inplace=True)
        return X
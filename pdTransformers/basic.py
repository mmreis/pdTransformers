from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Selects a subset of features of the provided data frame

    Parameters
    ----------
    attribute_names: a list of columns

    Examples
    --------
    """

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


class CreateDummies(BaseEstimator, TransformerMixin):
    def __init__(self, c_var, drop_original=False):
        self.c_var = c_var
        self.drop_original = drop_original
        self._columns = None

    def fit(self, X, y=None):
        for var in self.c_var:
            X = pd.concat([X, pd.get_dummies(X[var], prefix=var)], axis=1)
        if self.drop_original:
            X.drop(self.c_var, axis=1, inplace=True)
        self._columns = X.columns
        return self

    def transform(self, X, y=None):
        for var in self.c_var:
            X = pd.concat([X, pd.get_dummies(X[var], prefix=var)], axis=1)
        if self.drop_original:
            X.drop(self.c_var, axis=1, inplace=True)

        # see created columns and compare with columns that are supposed to be in X
        # fill with zeros the variables that aren't present in X
        for col in self._columns.difference(X.columns):
            X[col] = 0
        X = X[self._columns]
        return X


class Categorize(BaseEstimator, TransformerMixin):
    """
    Creates categories for a continuous variables (columns).

    Parameters
    ----------
    c_var: a list of columns
    nc_var: an array of same size that c_var, specifies the number of bins to consider.
    """
    def __init__(self, c_var, nc_var):
        self.c_var = c_var
        self.nc_var = nc_var
        # check if c_var and nc_var have the same length

    def fit(self, X, y=None):
        return self  # nothing else do do

    def transform(self, X, y=None):
        for var, n in zip(self.c_var, self.nc_var):
            X[var] = pd.cut(X[var], n).cat.codes
        return X


class DropNA(BaseEstimator, TransformerMixin):
    """
    Drops NaNs on the DataFrame
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.dropna(inplace=True)
        return X

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import *
import pandas as pd


class SetIndex(BaseEstimator, TransformerMixin):
    """
    Set column of the pandas data frame to Index
    """

    def __init__(self, index_name):
        self.index_name = index_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.index_name not in X.columns:
            ValueError("X columns do not contain '{}'".format(self.index_name))
        X.set_index(self.index_name, inplace=True)
        return X


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Selects a subset of features of the provided data frame

    Parameters
    ----------
    attribute_names: a list of columns

    Examples
    --------
            dataset = generate_ts(n_samples=1000, n_features=2, n_targets=1,
                                    split_X_y=False, start_date='2016-01-03 00:00',
                                    freq='1H')
            c = DataFrameSelector('target').fit_transform(dataset)

    """

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]

    def inverse_transform(self, X, y=None):
        return X


class CreateDummies(BaseEstimator, TransformerMixin):
    """
    Creates Dummies variables over a desired column or column set.

    :param c_var: list, data frame columns where the Dummies variables are to be created
    :param drop_original: bool (default=True) if True the original column is drooped from the returned data set

    Example:
            from WWIR.pd_transformers.basic import CreateDummies
            from WWIR.pd_transformers.time import AddDateTimeFeatures
            from sklearn.pipeline import Pipeline
            dataset = generate_ts(n_samples=1000, n_features=2, n_targets=1,
                                  split_X_y=False, start_date='2016-01-03 00:00',
                                  freq='1H')
            c_ = Pipeline([
                ('add_time', AddDateTimeFeatures({'hour': '%H'})),
                ('dummies', CreateDummies(c_var=['hour'], drop_original=True))])
            c_.fit_transform(dataset)

    """

    def __init__(self, c_var, drop_original=True):

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

    def inverse_transform(self, X, y=None):
        return X


class Categorize(BaseEstimator, TransformerMixin):
    """
    Creates categories for a continuous variables (columns).

    Parameters
    ----------
    :param c_var:: a list of columns
    :param nc_var:  a list of same size that c_var, specifies the number of bins to consider.

    Example:

    """

    def __init__(self, c_var, nc_var):
        self.c_var = c_var
        self.nc_var = nc_var
        # check if c_var and nc_var have the same length
        if len(c_var) != len(nc_var) and len(nc_var) == 1:
            self.nc_var = [nc_var for i in c_var]

    def fit(self, X, y=None):
        return self  # nothing else do do

    def transform(self, X, y=None):
        for var, n in zip(self.c_var, self.nc_var):
            X[var] = pd.cut(X[var], n).cat.codes
        return X

    def inverse_transform(self, X, y=None):
        return X


class DropNA(BaseEstimator, TransformerMixin):
    """
    Drops NaNs on the DataFrame
    """

    def __init__(self, axis=0):
        self.axis = axis
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.dropna(inplace=True, axis=self.axis)
        return X

    def inverse_transform(self, X, y=None):
        return X


class ScaleColumns(BaseEstimator, TransformerMixin):
    """
    Scales Columns based on a provided Transformer or Imputation Method

    :param f: customizable transformer, all skleran are available, should have fit and transform methods
    :param columns: list, columns where the transformation should occur
    :param f_args: dict, additional arguments for the f function

    Example:
            from WWIR.pd_transformers.basic import ScaleColumns
            from WWIR.pd_transformers.datasets import generate_ts

            dataset = generate_ts(n_samples=1000, n_features=2, n_targets=1,
            split_X_y=False, start_date='2016-01-03 00:00',
            freq='1H')

            dataset = ScaleColumns(f = 'MinMaxScaler', columns=[0, 1]).fit_transform(dataset)
    """

    def __init__(self, f, columns=None, f_args=None):
        self.columns = columns
        self.f_args = {} if f_args is None else f_args

        # if f is a string check if it is available in sklearn
        from sklearn.preprocessing import __all__ as list_of_available
        if isinstance(f, str) and f in list_of_available:
            self.f = eval(f + "(**%s)" % self.f_args.__str__())
        elif isinstance(f, str):
            raise ValueError(
                "f='{}' is not available in the sklearn.".format(f))
        elif hasattr(f, 'fit') and hasattr(f, 'transform'):
            # check if it has fit and transform methods
            self.f = f
        else:
            raise ValueError(
                "The provided transformer does not have fit and transform methods")

    def fit(self, X, y=None):
        if self.columns is None:
            self.f.fit(X)
        else:
            self.f.fit(X[self.columns])
        return self

    def transform(self, X, y=None):
        if self.columns is None:
            X[self.columns] = self.f.transform(X)
        else:
            X[self.columns] = self.f.transform(X[self.columns])
        return X

    def inverse_transform(self, X, y=None):
        if self.columns is None:
            X[self.columns] = self.f.inverse_transform(X)
        else:
            X[self.columns] = self.f.inverse_transform(X[self.columns])
        return X


class DataFrameSelectorByCategory(BaseEstimator, TransformerMixin):
    """
    Selects a subset of desired features filtering the examples by a given category

    Parameters
    ----------
    :param attribute_names: a list of columns
    :param cat: column to perform the filtering
    :param vcat: value used for filtering

    """

    def __init__(self, attribute_names, cat, vcat):
        self.attribute_names = attribute_names
        self.catg = cat
        self.vcatg = vcat

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X[X[self.catg] == self.vcatg]
        return X[self.attribute_names]


class DropColumns(BaseEstimator, TransformerMixin):
    """
    Drops Columns from Data Frame

    :param cols: str or list, column name for deletion
    """

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(list(set(self.cols).intersection(set(X.columns))), axis=1)

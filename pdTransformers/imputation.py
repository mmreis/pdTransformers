from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import *
import pandas as pd
import numpy as np
import numpy.ma as ma
import random
from sklearn.preprocessing import Imputer
from sklearn.utils import check_array
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
import warnings


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask


class SimpleImputation(BaseEstimator, TransformerMixin):
    """
    Imputation transformer for completing missing values with Simple Strategies

    :param missing_values: integer or "NaN", optional (default="NaN")
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For missing values encoded as np.nan,
        use the string value "NaN".
    :param strategy: string, optional (default="mean")
        Possible methods includes : "mean", "median", "most_frequent", "random", "replace"
    :param copy: integer, optional (default=0)
        The axis along which to impute.
    :param imputer_args: dict
        Additional arguments necessary for the chosen strategy

    """

    def __init__(self, missing_values='NaN', strategy='mean', copy=True, imputer_args=None):
        self.missing_values = missing_values
        self.strategy = strategy
        self.copy = copy
        self.imputer_args = {} if imputer_args is None else imputer_args
        self.axis = 0
        self.__object = None

    def fit(self, X, y=None):

        allowed_strategies = ["mean", "median", "most_frequent", "random", "replace"]
        X = check_array(X, accept_sparse='csc', dtype=np.float64,
                        force_all_finite=False)
        if self.strategy in ['mean', 'median', 'most_frequent']:
            self.__object = Imputer(missing_values=self.missing_values,
                                    strategy=self.strategy, **self.imputer_args).fit(X, y)

        elif self.strategy == 'random':
            self.__object = Imputer()
            self.__object.statistics_ = self._fit_random(X)

        elif self.strategy == 'replace':
            # check of replace is present in the provided arguments
            if 'new_values' not in self.imputer_args:
                raise ValueError("For strategy: '{0}' 'new_values' parameter "
                                 "should be provided".format(self.strategy))

            self.__object = Imputer()
            self.__object.statistics_ = self._fit_replace(X)

        else:
            raise ValueError("Can only use these strategies: {0} "
                             " got strategy={1}".format(allowed_strategies,
                                                        self.strategy))

        return self

    def transform(self, X):
        XX = pd.DataFrame(self.__object.transform(X), index=X.index, columns=X.columns)
        return XX

    def _fit_random(self, X):
        # compute self.statistics_
        mask = _get_mask(X, self.missing_values)
        masked_X = ma.masked_array(X, mask=mask)
        random_ = masked_X[np.random.choice(masked_X.shape[0], 1)].data[0]

        return random_

    def _fit_replace(self, X):

        val = np.array(self.imputer_args['new_values'])
        if val.size == X.shape[1]:
            return val.reshape((X.shape[1],))
        if val.size == 1:
            return np.repeat(val, X.shape[1])
        else:
            raise ValueError("'new_values' should be the same size as X axis = 1")


class ImputationWithEstimator(BaseEstimator, TransformerMixin):
    """
    Imputation transformer for completing missing values with
    an estimator (must be sklearn type with fit and predict methods implemented)
    :param base_inputs: str or list,
        Columns of the data frame without NaN's used to fit. Advised to use calendar variables
    :param impute_inputs: str or list,
        Columns in which to perform imputation, If not provided all columns except base_inputs are
        considered.
    :param estimator: estimator
        sklearn type with fit and predict methods implemented
    :param missing_values: integer or "NaN", optional (default="NaN")
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.
    :param n_jobs: int (default=1)
        number of process to use
    """

    def __init__(self, base_inputs, impute_inputs=None, estimator=LinearRegression(), missing_values="NaN", n_jobs=1):

        self.estimator = estimator
        self.n_jobs = n_jobs
        self.__ndim_y = None
        self.base_inputs = base_inputs
        self.missing_values = missing_values
        self.impute_inputs = impute_inputs

    def fit(self, X, y=None, sample_weight=None):
        XX = X.copy()
        XX.dropna(inplace=True)
        if self.impute_inputs is None:
            self.impute_inputs = XX.columns.difference(self.base_inputs)
        else:
            # check if inputs are available
            self.impute_inputs = list(set(self.impute_inputs).intersection(set(XX.columns)))
        self.base_inputs = list(set(self.base_inputs).intersection(set(XX.columns)))
        XX, yy = check_X_y(XX.loc[:, self.base_inputs], XX.loc[:, self.impute_inputs],
                           multi_output=True,
                           accept_sparse=True)
        self.__ndim_y = (yy.size == yy.shape[0])
        if self.__ndim_y == 1:
            self.estimators_ = clone(self.estimator)
            self.estimators_.fit(X=XX, y=yy, sample_weight=sample_weight)
        else:
            self.estimators_ = MultiOutputRegressor(estimator=self.estimator,
                                                    n_jobs=self.n_jobs). \
                fit(XX, yy, sample_weight=sample_weight)
        return self

    def transform(self, X):

        XX = X.copy()
        # c_base_inputs = XX.columns.difference(self.base_inputs)
        # if nan's are found in the base_input variables raise error
        if pd.isnull(XX[self.base_inputs]).any().any():
            ValueError("Input base variables should not contain NaN.")

        mask = _get_mask(XX.loc[:, self.impute_inputs], self.missing_values)
        # which rows have nan's
        row_with_nan = pd.isnull(XX).any(1).nonzero()[0]
        if len(row_with_nan) == 0:
            warnings.warn("Imputation wasn't necessary!")
            return XX
        preds_to_fill = np.zeros((mask.shape))
        preds_to_fill[row_with_nan, :] = self.estimators_.predict(XX.loc[XX.index[row_with_nan], self.base_inputs])
        XX = XX.fillna(0)

        XX[self.impute_inputs] = XX[self.self.impute_inputs] + mask * preds_to_fill

        return XX

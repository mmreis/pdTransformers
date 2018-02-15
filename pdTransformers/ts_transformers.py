# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.preprocessing as pp
import warnings

warnings.filterwarnings("ignore")

__numeric_f_time__ = ["%d", "%H", "%I", "%j", "%m", "%M", "%S", "%U", "%w", "%W", "%y"]


class SetIndex(BaseEstimator, TransformerMixin):
    """
    Set column of the pandas data frame to Index
    """
    def __init__(self, index_name):
        self.index_name = index_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.set_index(self.index_name, inplace=True)
        return X





class DataFrameSelectorCateg(BaseEstimator, TransformerMixin):
    """
    Selects a subset of desired features filtering the examples by a given category

    Parameters
    ----------
    attribute_names: a list of columns
    cat: column to perform the filtering
    vcat: value used for filtering

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


class AddVariabilityIndexes(BaseEstimator, TransformerMixin):
    def __init__(self, col, n, m):
        self.column = col
        self.n = n
        self.m = m

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self


class AddLaggedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, lag_vars):
        self.lag_vars = lag_vars

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        new_dict = {}
        for predictor, all_lags in self.lag_vars.items():
            if predictor not in X.columns:
                print(' ## ERROR! Lags from \'{}\' were excluded, since the dataset wasn\'t loaded.'.format(predictor))
                continue
            new_dict[predictor] = X[predictor]
            for l in all_lags:
                new_dict['%s_lag%d' % (predictor, l)] = X[predictor].shift(l)
                # if index_lag:
                #     new_dict['index_lag%d' % l] = df['predy'].shift(l)
                # res = pd.DataFrame(new_dict, index=X.index)



class DeTrendPoly(BaseEstimator, TransformerMixin):
    def __init__(self, deg, var, copy=True, **kwargs):
        self.deg = deg
        self.var = var
        self._kwargs = kwargs
        self._coef = []
        self._trend = []

    def fit(self, X, y=None):
        # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
        #                 warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES)
        self._coef = np.polyfit(x=np.arange(0, X.shape[0]), y=X[self.var].values, deg=self.deg, **self._kwargs)
        # self._coef = [np.polyfit(x=np.arange(0, X.shape[0]), y=X[:, i], deg=self.deg, **self._kwargs) for i in
        #          range(X.shape[1])]
        return self

    def transform(self, X):
        # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
        #                 warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES)
        self._trend = np.polyval(p=self._coef, x=np.arange(0, X.shape[0]))
        X[self.var] = X[self.var] - self._trend
        return X

    def inverse_transform(self, X):
        X[self.var] = X[self.var] + self._trend


class DeTrendOp3(BaseEstimator, TransformerMixin):
    """
    name: the variable name you want to create
    var: variable in with to compute the transformation
    var_lower: variable in with to compute the transformation (lower "bound")
    var_upper: variable in with to compute the transformation (upper "bound")
    """

    def __init__(self, name, var, var_lower, var_upper):
        self.name = name
        self.var = var
        self.var_lower = var_lower
        self.var_upper = var_upper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.name] = X[self.var] * (X[self.var_upper] - X[self.var_lower]) + X[self.var_lower]
        return X


class DeTrendOp2(BaseEstimator, TransformerMixin):
    """
    name: the variable name you want to create
    var: variable in with to compute the transformation
    var_lower: variable in with to compute the transformation (lower "bound")
    var_upper: variable in with to compute the transformation (upper "bound")
    """

    def __init__(self, name, var, var_lower, var_upper):
        self.name = name
        self.var = var
        self.var_lower = var_lower
        self.var_upper = var_upper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.name] = (X[self.var] - X[self.var_lower]) / (X[self.var_upper] - X[self.var_lower])
        return X


class DeTrendOp1(BaseEstimator, TransformerMixin):
    def __init__(self, name, var, by):
        self.name = name
        self.var = var
        self.by = by

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.name] = (X[self.var] - X[self.by])
        return X


class InsertAggByTimeLags(BaseEstimator, TransformerMixin):
    def __init__(self, lags, timev='index'):
        self.lags = lags
        self.timev = timev

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # @todo add exeption in case self.None where the lags are created for every columns

        # check self.timev is a date column type
        # if not print error
        XX = X.copy()
        # if XX.index.name == self.timev:
        XX.reset_index(inplace=True)
        if self.timev == 'index':
            self.timev = X.index.name

        if hasattr(X, 'columns'):
            original_cols = X.columns.values.tolist()
            subs = [item for item in self.lags.keys()]
            original_cols = [i for i, item in enumerate(original_cols) if item in subs]
        else:  # @todo add exeption for array
            original_cols = list(range(len(X[0, :])))

        for predictor, all_lags in self.lags.items():
            if predictor not in X.columns:
                print(' ## ERROR! Lags from \'{}\' were excluded, since the dataset wasn\'t loaded.'.format(predictor))
                continue

            # filter dataset with predictor and time_v

            # print(predictor)

            df = XX[[self.timev, predictor]]

            for tuple_ in all_lags:
                agg, by, lags = tuple_

                if by.lower() in ['hour', 'h']:
                    XX['ts'] = XX[self.timev].values.astype('<M8[h]')

                    df['ts'] = XX[self.timev].values.astype('<M8[h]')
                    df_agg = eval("df.groupby(['ts'])." + agg + "()")

                    if isinstance(lags, list):
                        for lag in lags:
                            XX['ts_lag'] = (XX['ts'] - pd.DateOffset(hours=lag))
                            XX = XX.merge(df_agg, left_on='ts_lag', right_index=True, how='left',
                                          suffixes=('', '_%s_%s_%d' % (agg.lower(), by.lower(), lag)))
                    else:
                        XX['ts_lag'] = (XX['ts'] - pd.DateOffset(hours=lags))
                        XX = XX.merge(df_agg, left_on='ts_lag', right_index=True, how='left',
                                      suffixes=('', '_%s_%s_%d' % (agg.lower(), by.lower(), lags)))

                elif by.lower() in ['week', 'w']:
                    if 'week_year' not in XX.columns:
                        XX['ts'] = XX.set_index(self.timev).index.strftime('%U-%Y')
                    else:
                        XX['ts'] = XX['week_year']

                    df['ts'] = XX['ts']
                    df_agg = eval("df.groupby(['ts'])." + agg + "()")

                    if isinstance(lags, list):
                        for lag in lags:
                            XX['ts_lag'] = (XX[self.timev] - pd.DateOffset(weeks=lag)).dt.strftime('%U-%Y')
                            XX = XX.merge(df_agg, left_on='ts_lag', right_index=True, how='left',
                                          suffixes=('', '_%s_%s_%d' % (agg.lower(), by.lower(), lag)))
                    else:
                        XX['ts_lag'] = (XX[self.timev] - pd.DateOffset(weeks=lags)).dt.strftime('%U-%Y')
                        XX = XX.merge(df_agg, left_on='ts_lag', right_index=True, how='left',
                                      suffixes=('', '_%s_%s_%d' % (agg.lower, by.lower(), lags)))

                    if 'ts' in XX.columns:
                        XX.drop('ts', axis=1, inplace=True)
                    if 'ts_lag' in XX.columns:
                        XX.drop('ts_lag', axis=1, inplace=True)
                else:
                    XX['ts'] = XX.set_index(self.timev).index.floor(by)

                    df['ts'] = XX['ts']
                    df_agg = eval(
                        "df.set_index('" + self.timev + "')['" + predictor + "'].resample('" + by + "')." + agg + "()")

                    if isinstance(lags, list):
                        for lag in lags:
                            XX['ts_lag'] = (XX['ts'] - pd.Timedelta(by))  # @todo lag should be hourly
                            XX = XX.merge(df_agg.to_frame(), left_on='ts_lag', right_index=True, how='left',
                                          suffixes=('', '_%s_%s_%d' % (agg.lower(), by.lower(), lag)))

                    else:
                        XX['ts_lag'] = (XX['ts'] - pd.Timedelta(by))
                        XX = XX.merge(df_agg.to_frame(), left_on='ts_lag', right_index=True, how='left',
                                      suffixes=('', '_%s_%s_%d' % (agg.lower, by.lower(), lags)))

            if 'ts' in XX.columns:
                XX.drop('ts', axis=1, inplace=True)
            if 'ts_lag' in XX.columns:
                XX.drop('ts_lag', axis=1, inplace=True)

        XX.set_index(self.timev, inplace=True)
        return XX


class InsertChangeOverTime(BaseEstimator, TransformerMixin):
    def __init__(self, predictors, lag=1):
        self.predictors = predictors
        self.lag = lag

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X['ts'] = X.index
        aux = pd.DatetimeIndex(X.ts - X.ts.shift(self.lag).fillna(0))
        X['dftime'] = aux.hour * 60 + aux.minute

        if isinstance(self.predictors, list):
            for variable in self.predictors:
                # change over time
                X['%s_C' % variable] = X[variable] - X[variable].shift(self.lag).fillna(0)
                X['%s_CoT' % variable] = X['%s_C' % variable] / X['dftime']

                # rate of change over time
                X['%s_RT' % variable] = (X['%s_CoT' % variable] - X['%s_CoT' % variable].shift(self.lag)) / X['dftime']

                # growth or decay
                X['%s_GorD' % variable] = X['%s_C' % variable] / X[variable]

                X.drop('%s_C' % variable, axis=1, inplace=True)
        else:
            variable = self.predictors
            X['%s_C' % variable] = X[variable] - X[variable].shift(self.lag).fillna(0)
            X['%s_CoT' % variable] = X['%s_C' % variable] / X['dftime']

            # rate of change over time
            X['%s_RT' % variable] = (X['%s_CoT' % variable] - X['%s_CoT' % variable].shift(self.lag)) / X['dftime']

            # growth or decay
            X['%s_GorD' % variable] = X['%s_C' % variable] / X[variable]

            X.drop('%s_C' % variable, axis=1, inplace=True)

        if 'ts' in X.columns:
            X.drop('ts', axis=1, inplace=True)
        if 'dftime' in X.columns:
            X.drop('dftime', axis=1, inplace=True)

        return X


class Filter4Lag(BaseEstimator, TransformerMixin):
    def __init__(self, tol):
        self.tol = tol

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n_original = X.shape[0]
        X['tvalue'] = X.index
        X['delta'] = (X['tvalue'] - X['tvalue'].shift()).fillna(0)
        X['diff_delta'] = X['delta'].apply(lambda x: x / np.timedelta64(1, 'm')).astype('int64') % (24 * 60)
        X = X[X.diff_delta <= self.tol]
        X.drop(['tvalue', 'delta', 'diff_delta'], axis=1, inplace=True)
        n_final = X.shape[0]
        print("%f '%s of data lost with filter" % ((1 - n_final / n_original) * 100, '%'))

        return X


class DataFrameCompute(BaseEstimator, TransformerMixin):
    def __init__(self, what="corr", transpose=False, fill_na=None):
        self.what = what
        # todo: convert to function
        self.transpose = transpose
        self.fill_na = fill_na

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.transpose:
            XX = eval("X.T." + self.what + "()")
        else:
            XX = eval("X." + self.what + "()")
        if self.fill_na is not None:
            return XX.fillna(self.fill_na)
        else:
            return XX


class AddPolynomialFeatures(BaseEstimator, TransformerMixin):
    """Generate polynomial and interaction features.

    Generate a new feature matrix consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.
    For example, if an input sample is two dimensional and of the form
    [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    Parameters
    ----------
    degree : integer
        The degree of the polynomial features. Default = 2.

    interaction_only : boolean, default = False
        If true, only interaction features are produced: features that are
        products of at most ``degree`` *distinct* input features (so not
        ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).

    include_bias : boolean
        If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).

    exclude_feat : list, default = None
        list of features to exclude from the polynomial feature creation
    """

    def __init__(self, degree=2, interaction_only=False, include_bias=True, exclude_feat=None):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.exclude_feat = exclude_feat
        self._t = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # remove object attributes
        XX = X.select_dtypes(exclude=['object']).copy()
        # remove extract_feat
        XX = XX[XX.columns.difference([self.exclude_feat])]
        XX_c = X.select_dtypes(include=['object'])
        XX_c = pd.concat([XX_c, X[self.exclude_feat]], axis=1)
        self._t = pp.PolynomialFeatures(degree=self.degree,
                                        interaction_only=self.interaction_only,
                                        include_bias=self.include_bias)
        self._t.fit(XX)
        XX = pd.DataFrame(self._t.transform(XX), columns=self._t.get_feature_names(XX.columns), index=X.index)
        XX = pd.concat([XX, XX_c], axis=1)
        return XX


class TimeSeriesGroupBy(BaseEstimator, TransformerMixin):
    def __init__(self, by, agg={'mean'}):
        self.by = by
        self.agg = agg
        self.fitX_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.fitX_ = X.groupby(self.by).agg(self.agg)
        self.fitX_.columns = ['__'.join(col) for col in self.fitX_.columns]
        self.fitX_ = self.fitX_.T
        return self.fitX_






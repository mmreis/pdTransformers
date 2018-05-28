# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone
import sklearn.preprocessing as pp
import warnings

warnings.filterwarnings("ignore")

__numeric_f_time__ = ["%d", "%H", "%I", "%j", "%m", "%M", "%S", "%U", "%w", "%W", "%y"]


class InsertDifferences(BaseEstimator, TransformerMixin):
    """
    Creates columns with time series differences

    :param lags: dict,
        dictionary with key as the variable (column) in which to perform differences

    Example:
        from WWIR.pd_transformers.datasets import generate_ts
        dataset = generate_ts(n_samples=1000, n_features=2, n_targets=1,
                              split_X_y=False, start_date='2016-01-03 00:00',
                              freq='1H')
        from WWIR.pd_transformers.ts_transformers import InsertDifferences
        ct = InsertDifferences(lags={'target': [1]})
        ct.fit_transform(dataset)
    """

    def __init__(self, lags):
        self.lags = lags

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if hasattr(X, 'columns'):
            original_cols = X.columns.values.tolist()
            subs = [item for item in self.lags.keys()]
            original_cols = [i for i, item in enumerate(original_cols) if item in subs]
        else:
            original_cols = list(range(len(X[0, :])))

        new_dict = {}
        for predictor, periods in self.lags.items():
            if predictor not in X.columns:
                warnings.warn(' ## ERROR! Lags from \'{}\' were excluded, since the dataset wasn\'t loaded.'.format(predictor))
                continue
            new_dict[predictor] = X[predictor]
            for l in periods:
                # new_dict['%s_lag%d' % (predictor, l)] = X[predictor].shift(l)
                X['%s_diff-%d' % (predictor, l)] = X[predictor].diff(l)
        return X


class AddVariabilityIndexes(BaseEstimator, TransformerMixin):
    """
    Variability Indexes for Time Series

    :param col: str, column name
    :param n: int, number of observations to smoothing
    :param m: int, window of width
    :param quantile_range: 2D-tuple, desired inter-quantile range

    Example:

    References:
        Anastasiades, G., & McSharry, P. (2013).
        Quantile forecasting of wind power using variability indices.
        Energies, 6(2), 662–695. https://doi.org/10.3390/en6020662
    """

    def __init__(self, col, n, m, quantile_range=(0.25, 0.75)):
        self.column = col
        self.n = n
        self.m = m
        self.quantile_range = quantile_range

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        m = self.m
        n = self.n
        XX = X.copy()
        A = XX[self.column]
        R = A.rolling(window=self.m + 1, center=False).mean()
        R.iloc[:m] = R.iloc[m]
        SD = R.rolling(window=n + 1, center=False).std()
        SD.iloc[:n] = SD.iloc[n]
        QMIN = R.rolling(window=n + 1, center=False).quantile(self.quantile_range[0])
        QMIN.iloc[:n] = QMIN.iloc[n]
        QMAX = R.rolling(window=n + 1, center=False).quantile(self.quantile_range[1])
        QMAX.iloc[:n] = QMAX.iloc[n]

        XX['R'] = R
        XX['SD'] = SD
        XX['Q%s' % str(int(self.quantile_range[1] * 100)).zfill(2)] = QMAX
        XX['Q%s' % str(int(self.quantile_range[0] * 100)).zfill(2)] = QMIN
        XX['%s_IQR' % self.column] = QMAX - QMIN
        return XX


class InsertChangeOverTime(BaseEstimator, TransformerMixin):
    """
    Constructs features based on change and rate of change (essentially the lag
    differences) and percentage changes (growth or decay).

    :param predictors: list or str, column(s) name
    :param lag: int, number of lags to first differences

    Example:
            from WWIR.pd_transformers.datasets import generate_ts
            dataset = generate_ts(n_samples=1000, n_features=2, n_targets=1,
                              split_X_y=False, start_date='2016-01-03 00:00',
                              freq='1H')
            from WWIR.pd_transformers.ts_transformers import InsertChangeOverTime

    References:
    https://www.analyticsvidhya.com/blog/2017/04/feature-engineering-in-iot-age-how-to-deal-with-iot-data-and-create-features-for-machine-learning/

    """

    def __init__(self, predictors, lag=1):
        self.predictors = predictors
        self.lag = lag

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.sort()
        X['ts'] = X.index
        aux = pd.DatetimeIndex((X.ts - X.ts.shift(self.lag)).fillna(0))
        X['dftime'] = aux.hour * 60 + aux.minute
        X['dftime'] = X['dftime'].clip(1)

        predi = self.predictors if isinstance(self.predictors, list) else [self.predictors]

        for variable in predi:
            # change over time
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
        return X.bfill()

    def inverse_transform(self, X, y=None):
        return X


class InsertAggByTimeLags(BaseEstimator, TransformerMixin):
    """

    Insert lags using date to date correspondence with aggregated lagged values

    Example:
            from WWIR.pd_transformers.datasets import generate_ts
            dataset = generate_ts(n_samples=1000, n_features=2, n_targets=1,
                                  split_X_y=False, start_date='2016-01-03 00:00',
                                  freq='1H')
            from WWIR.pd_transformers.ts_transformers import InsertAggByTimeLags as IATL
            c = IATL(agg_lags={'target': [('mean', '5min', ['24H'])]})
            c.fit_transform(dataset)

    """

    def __init__(self, agg_lags, timev=None):
        # self.lags = [agg_lags] if isinstance(agg_lags, dict) else agg_lags
        self.lags = agg_lags
        self.timev = timev

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        XX = X.copy()
        ind_name = 'index' if XX.index.name is None else XX.index.name
        ind_freq = XX.index.freqstr
        XX.reset_index(inplace=True)

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

            df = XX[[ind_name, predictor]].set_index(ind_name)

            for tuple_ in all_lags:
                agg, by, lags = tuple_

                df_agg = df.resample(by).agg(agg)
                df_agg['ts'] = df_agg.index
                XX['ts'] = df_agg.resample(ind_freq).ffill()['ts'][XX['index']].values

                if isinstance(df_agg, pd.Series):
                    df_agg = df_agg.to_frame()

                lags = lags if isinstance(lags, list) else [lags]

                if isinstance(lags, list):
                    for lag in lags:
                        try:
                            XX['ts_lag'] = XX['ts'] - pd.Timedelta(lag)
                        except:
                            try:
                                XX['ts_lag'] = XX['ts'] - pd.tseries.frequencies.to_offset(lag)
                                # todo: find cheaper solution
                            except:
                                print('if this prints, we had no error!')  # won't print!

                        XX = pd.merge(XX, df_agg, left_on="ts_lag", right_on="ts", how="left",
                                      suffixes=('', '_%s_%s_%s' % (agg, by, lag)))
                        XX.drop('ts_%s_%s_%s' % (agg, by, lag), axis=1, inplace=True)

            XX.drop(['ts_lag', 'ts'], axis=1, inplace=True)
        XX.set_index(ind_name, inplace=True)
        return XX.sort()

    def inverse_transform(self, X, y=None):
        return X


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

# class Filter4Lag(BaseEstimator, TransformerMixin):
#     def __init__(self, tol):
#         self.tol = tol
#         # self.__fitted = 0
#
#     def fit(self, X, y=None):
#         # self.__fitted = 1
#         return self
#
#     def transform(self, X, y=None):
#         # if self.__fitted == 2:
#         #     return X
#         n_original = X.shape[0]
#         X['tvalue'] = X.index
#         X['delta'] = (X['tvalue'] - X['tvalue'].shift()).fillna(0)
#         X['diff_delta'] = X['delta'].apply(lambda x: x / np.timedelta64(1, 'm')).astype('int64') % (24 * 60)
#         X = X[X.diff_delta <= self.tol]
#         X.drop(['tvalue', 'delta', 'diff_delta'], axis=1, inplace=True)
#         n_final = X.shape[0]
#         print("%f '%s of data lost with filter" % ((1 - n_final / n_original) * 100, '%'))
#         # self.__fitted = 2
#         return X
#
#     def inverse_transform(self, X, y=None):
#         return X

#
# class DataFrameCompute(BaseEstimator, TransformerMixin):
#     def __init__(self, what="corr", transpose=False, fill_na=None):
#         self.what = what
#         # todo: convert to function
#         self.transpose = transpose
#         self.fill_na = fill_na
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         if self.transpose:
#             XX = eval("X.T." + self.what + "()")
#         else:
#             XX = eval("X." + self.what + "()")
#         if self.fill_na is not None:
#             return XX.fillna(self.fill_na)
#         else:
#             return XX
#
#

#
#
# class TimeSeriesGroupBy(BaseEstimator, TransformerMixin):
#     def __init__(self, by, agg={'mean'}):
#         self.by = by
#         self.agg = agg
#         self.fitX_ = None
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         self.fitX_ = X.groupby(self.by).agg(self.agg)
#         self.fitX_.columns = ['__'.join(col) for col in self.fitX_.columns]
#         self.fitX_ = self.fitX_.T
#         return self.fitX_
#
#
# class TransformByColumnCategory(BaseEstimator, TransformerMixin):
#     """
#     cat column in which the categories are present
#     transformer
#     col_t columns in which the transformer will be applied
#     """
#
#     def __init__(self, cat, transformer=pp.StandardScaler(), col_t=None):
#         self.cat = cat
#         self.transformer = transformer
#         self._transformers = None
#         self._c = [col_t] if isinstance(col_t, str) else col_t
#         self.col_t = col_t
#
#     def fit(self, X, y=None):
#         # computes the number of unique categories cat has
#
#         if self._c is not None:
#             if self.cat not in self._c:
#                 self._c.append(self.cat)
#             X = X[self._c]
#         # nc = X[self.cat].unique().__len__()
#         c = X[self.cat].unique()
#         # initializes the nc transformers need
#         self._transformers = dict(zip(c, [clone(self.transformer) for i in c]))
#         for i in c:
#             self._transformers[i].fit(X[X[self.cat] == i].drop(self.cat, axis=1))
#
#         return self
#
#     def transform(self, X):
#         if self.col_t is not None:
#             XX = X[self._c].copy()
#         else:
#             XX = X.copy()
#         XXX = pd.DataFrame()
#         nnp_flag = 0
#         for i in self._transformers.keys():
#             res = self._transformers[i].transform(XX[XX[self.cat] == i].drop(self.cat,
#                                                                              axis=1))
#             if isinstance(res, np.ndarray):
#                 XX.loc[XX[self.cat] == i, self.col_t] = res
#             else:
#                 nnp_flag = 1
#                 XXX = pd.concat([res, XXX])
#
#         if nnp_flag == 1:
#             XX = XXX.copy()
#             del XXX
#             X = X.ix[XX.index]
#
#         if self.col_t is not None:
#             X[self.col_t] = XX[self.col_t]
#             return X
#         else:
#             return XX
#
#     def inverse_transform(self, X):
#         if self.col_t is not None:
#             XX = X[self._c].copy()
#         else:
#             XX = X.copy()
#         XXX = pd.DataFrame()
#         nnp_flag = 0
#         for i in self._transformers.keys():
#             res = self._transformers[i].inverse_transform(XX[XX[self.cat] == i].drop(self.cat,
#                                                                                      axis=1))
#             if isinstance(res, np.ndarray):
#                 XX.loc[XX[self.cat] == i, self.col_t] = res
#             else:
#                 nnp_flag = 1
#                 XXX = pd.concat([res, XXX])
#
#         if nnp_flag == 1:
#             XX = XXX.copy()
#             del XXX
#             X = X.ix[XX.index]
#
#         if self.col_t is not None:
#             X[self.col_t] = XX[self.col_t]
#             return X
#         else:
#             return XX
#
#
# class OutlierMAD(BaseEstimator, TransformerMixin):
#     def __init__(self, var, threshold=3):
#         self.threshold = threshold
#         self.var = var
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         XX = X[self.var]
#         if len(XX.shape) == 1:
#             XX = XX[:, None]
#         median = np.median(XX)
#         diff = np.sum((XX - median) ** 2, axis=-1)
#         diff = np.sqrt(diff)
#         d_median = np.median(diff)
#         if np.abs(d_median) < np.finfo(np.float32).eps:
#             d_median = 1.0
#         modified_z_score = 0.6745 * diff / d_median
#         print(np.median(diff))
#         out = modified_z_score > self.threshold
#         return X[~out]
#
#
# class DeTrendPoly(BaseEstimator, TransformerMixin):
#     def __init__(self, deg, var, copy=True, **kwargs):
#         self.deg = deg
#         self.var = var
#         self._kwargs = kwargs
#         self._coef = []
#         self._trend = []
#
#     def fit(self, X, y=None):
#         # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
#         #                 warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES)
#         self._coef = np.polyfit(x=np.arange(0, X.shape[0]), y=X[self.var].values, deg=self.deg, **self._kwargs)
#         # self._coef = [np.polyfit(x=np.arange(0, X.shape[0]), y=X[:, i], deg=self.deg, **self._kwargs) for i in
#         #          range(X.shape[1])]
#         return self
#
#     def transform(self, X):
#         # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
#         #                 warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES)
#         self._trend = np.polyval(p=self._coef, x=np.arange(0, X.shape[0]))
#         X[self.var] = X[self.var] - self._trend
#         return X
#
#     def inverse_transform(self, X):
#         X[self.var] = X[self.var] + self._trend
#
#
# class DeTrendOp3(BaseEstimator, TransformerMixin):
#     """
#     name: the variable name you want to create
#     var: variable in with to compute the transformation
#     var_lower: variable in with to compute the transformation (lower "bound")
#     var_upper: variable in with to compute the transformation (upper "bound")
#     """
#
#     def __init__(self, name, var, var_lower, var_upper):
#         self.name = name
#         self.var = var
#         self.var_lower = var_lower
#         self.var_upper = var_upper
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X[self.name] = X[self.var] * (X[self.var_upper] - X[self.var_lower]) + X[self.var_lower]
#         return X
#
#
# class DeTrendOp2(BaseEstimator, TransformerMixin):
#     """
#     name: the variable name you want to create
#     var: variable in with to compute the transformation
#     var_lower: variable in with to compute the transformation (lower "bound")
#     var_upper: variable in with to compute the transformation (upper "bound")
#     """
#
#     def __init__(self, name, var, var_lower, var_upper):
#         self.name = name
#         self.var = var
#         self.var_lower = var_lower
#         self.var_upper = var_upper
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X[self.name] = (X[self.var] - X[self.var_lower]) / (X[self.var_upper] - X[self.var_lower])
#         return X


# class DeTrendOp1(BaseEstimator, TransformerMixin):
#     def __init__(self, name, var, by):
#         self.name = name
#         self.var = var
#         self.by = by
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X[self.name] = (X[self.var] - X[self.by])
#         return X



# # To support both python 2 and python 3
# from __future__ import division, print_function, unicode_literals
#
# import pandas as pd
# import numpy as np
#
# from sklearn.base import BaseEstimator, TransformerMixin, clone
# import sklearn.preprocessing as pp
# import warnings
#
# warnings.filterwarnings("ignore")
#
# __numeric_f_time__ = ["%d", "%H", "%I", "%j", "%m", "%M", "%S", "%U", "%w", "%W", "%y"]
#
#
# class AddVariabilityIndexes(BaseEstimator, TransformerMixin):
#     """
#     Variability Indexes for Time Series
#
#     :param col: str, column name
#     :param n: int, number of observations to smoothing
#     :param m: int, window of width
#     :param quantile_range: 2D-tuple, desired inter-quantile range
#
#     Example:
#
#     References:
#         Anastasiades, G., & McSharry, P. (2013).
#         Quantile forecasting of wind power using variability indices.
#         Energies, 6(2), 662–695. https://doi.org/10.3390/en6020662
#     """
#
#     def __init__(self, col, n, m, quantile_range=(0.25, 0.75)):
#         self.column = col
#         self.n = n
#         self.m = m
#         self.quantile_range = quantile_range
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         m = self.m
#         n = self.n
#         XX = X.copy()
#         A = XX[self.column]
#         R = A.rolling(window=self.m + 1, center=False).mean()
#         R.iloc[:m] = R.iloc[m]
#         SD = R.rolling(window=n + 1, center=False).std()
#         SD.iloc[:n] = SD.iloc[n]
#         QMIN = R.rolling(window=n + 1, center=False).quantile(self.quantile_range[0])
#         QMIN.iloc[:n] = QMIN.iloc[n]
#         QMAX = R.rolling(window=n + 1, center=False).quantile(self.quantile_range[1])
#         QMAX.iloc[:n] = QMAX.iloc[n]
#
#         XX['R'] = R
#         XX['SD'] = SD
#         XX['Q%s' % str(int(self.quantile_range[1] * 100)).zfill(2)] = QMAX
#         XX['Q%s' % str(int(self.quantile_range[0] * 100)).zfill(2)] = QMIN
#         XX['%s_IQR' % self.column] = QMAX - QMIN
#         return XX
#
#
# class InsertChangeOverTime(BaseEstimator, TransformerMixin):
#     """
#     Constructs features based on change and rate of change (essentially the lag
#     differences) and percentage changes (growth or decay).
#
#     :param predictors: list or str, column(s) name
#     :param lag: int, number of lags to first differences
#
#     Example:
#             from WWIR.pd_transformers.datasets import generate_ts
#             dataset = generate_ts(n_samples=1000, n_features=2, n_targets=1,
#                               split_X_y=False, start_date='2016-01-03 00:00',
#                               freq='1H')
#             from WWIR.pd_transformers.ts_transformers import InsertChangeOverTime
#
#     References:
#     https://www.analyticsvidhya.com/blog/2017/04/feature-engineering-in-iot-age-how-to-deal-with-iot-data-and-create-features-for-machine-learning/
#
#     """
#
#     def __init__(self, predictors, lag=1):
#         self.predictors = predictors
#         self.lag = lag
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         X = X.sort()
#         X['ts'] = X.index
#         aux = pd.DatetimeIndex((X.ts - X.ts.shift(self.lag)).fillna(0))
#         X['dftime'] = aux.hour * 60 + aux.minute
#         X['dftime'] = X['dftime'].clip(1)
#
#         predi = self.predictors if isinstance(self.predictors, list) else [self.predictors]
#
#         for variable in predi:
#             # change over time
#             X['%s_C' % variable] = X[variable] - X[variable].shift(self.lag).fillna(0)
#             X['%s_CoT' % variable] = X['%s_C' % variable] / X['dftime']
#
#             # rate of change over time
#             X['%s_RT' % variable] = (X['%s_CoT' % variable] - X['%s_CoT' % variable].shift(self.lag)) / X['dftime']
#
#             # growth or decay
#             X['%s_GorD' % variable] = X['%s_C' % variable] / X[variable]
#
#             X.drop('%s_C' % variable, axis=1, inplace=True)
#
#         if 'ts' in X.columns:
#             X.drop('ts', axis=1, inplace=True)
#         if 'dftime' in X.columns:
#             X.drop('dftime', axis=1, inplace=True)
#         return X.bfill()
#
#     def inverse_transform(self, X, y=None):
#         return X
#
#
# class InsertAggByTimeLags(BaseEstimator, TransformerMixin):
#     """
#
#     Insert lags using date to date correspondence with aggregated lagged values
#
#     Example:
#             from WWIR.pd_transformers.datasets import generate_ts
#             dataset = generate_ts(n_samples=1000, n_features=2, n_targets=1,
#                                   split_X_y=False, start_date='2016-01-03 00:00',
#                                   freq='1H')
#             from WWIR.pd_transformers.ts_transformers import InsertAggByTimeLags as IATL
#             c = IATL(agg_lags={'target': [('mean', '5min', ['24H'])]})
#             c.fit_transform(dataset)
#
#     :param agg_lags:
#     :param timev:
#     """
#
#     def __init__(self, agg_lags, timev=None):
#         # self.lags = [agg_lags] if isinstance(agg_lags, dict) else agg_lags
#         self.lags = agg_lags
#         self.timev = timev
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#
#         XX = X.copy()
#         ind_name = 'index' if XX.index.name is None else XX.index.name
#         ind_freq = XX.index.freqstr
#         XX.reset_index(inplace=True)
#
#         if hasattr(X, 'columns'):
#             original_cols = X.columns.values.tolist()
#             subs = [item for item in self.lags.keys()]
#             original_cols = [i for i, item in enumerate(original_cols) if item in subs]
#         else:  # @todo add exeption for array
#             original_cols = list(range(len(X[0, :])))
#
#         for predictor, all_lags in self.lags.items():
#             if predictor not in X.columns:
#                 print(' ## ERROR! Lags from \'{}\' were excluded, since the dataset wasn\'t loaded.'.format(predictor))
#                 continue
#
#             df = XX[[ind_name, predictor]].set_index(ind_name)
#
#             for tuple_ in all_lags:
#                 agg, by, lags = tuple_
#
#                 df_agg = df.resample(by).agg(agg)
#                 df_agg['ts'] = df_agg.index
#                 XX['ts'] = df_agg.resample(ind_freq).ffill()['ts'][XX['index']].values
#
#                 if isinstance(df_agg, pd.Series):
#                     df_agg = df_agg.to_frame()
#
#                 lags = lags if isinstance(lags, list) else [lags]
#
#                 if isinstance(lags, list):
#                     for lag in lags:
#                         try:
#                             XX['ts_lag'] = XX['ts'] - pd.Timedelta(lag)
#                         except:
#                             try:
#                                 XX['ts_lag'] = XX['ts'] - pd.tseries.frequencies.to_offset(lag)
#                                 # todo: find cheaper solution
#                             except:
#                                 print('if this prints, we had no error!')  # won't print!
#
#                         XX = pd.merge(XX, df_agg, left_on="ts_lag", right_on="ts", how="left",
#                                       suffixes=('', '_%s_%s_%s' % (agg, by, lag)))
#                         XX.drop('ts_%s_%s_%s' % (agg, by, lag), axis=1, inplace=True)
#
#             XX.drop(['ts_lag', 'ts'], axis=1, inplace=True)
#         XX.set_index(ind_name, inplace=True)
#         return XX.sort()
#
#     def inverse_transform(self, X, y=None):
#         return X
#
#
# # class Filter4Lag(BaseEstimator, TransformerMixin):
# #     def __init__(self, tol):
# #         self.tol = tol
# #         # self.__fitted = 0
# #
# #     def fit(self, X, y=None):
# #         # self.__fitted = 1
# #         return self
# #
# #     def transform(self, X, y=None):
# #         # if self.__fitted == 2:
# #         #     return X
# #         n_original = X.shape[0]
# #         X['tvalue'] = X.index
# #         X['delta'] = (X['tvalue'] - X['tvalue'].shift()).fillna(0)
# #         X['diff_delta'] = X['delta'].apply(lambda x: x / np.timedelta64(1, 'm')).astype('int64') % (24 * 60)
# #         X = X[X.diff_delta <= self.tol]
# #         X.drop(['tvalue', 'delta', 'diff_delta'], axis=1, inplace=True)
# #         n_final = X.shape[0]
# #         print("%f '%s of data lost with filter" % ((1 - n_final / n_original) * 100, '%'))
# #         # self.__fitted = 2
# #         return X
# #
# #     def inverse_transform(self, X, y=None):
# #         return X
#
#
# class DataFrameSelectorCateg(BaseEstimator, TransformerMixin):
#     """
#     Selects a subset of desired features filtering the examples by a given category
#
#     Parameters
#     ----------
#     attribute_names: a list of columns
#     cat: column to perform the filtering
#     vcat: value used for filtering
#
#     """
#
#     def __init__(self, attribute_names, cat, vcat):
#         self.attribute_names = attribute_names
#         self.catg = cat
#         self.vcatg = vcat
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X = X[X[self.catg] == self.vcatg]
#         return X[self.attribute_names]
#
# class DeTrendPoly(BaseEstimator, TransformerMixin):
#     def __init__(self, deg, var, copy=True, **kwargs):
#         self.deg = deg
#         self.var = var
#         self._kwargs = kwargs
#         self._coef = []
#         self._trend = []
#
#     def fit(self, X, y=None):
#         # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
#         #                 warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES)
#         self._coef = np.polyfit(x=np.arange(0, X.shape[0]), y=X[self.var].values, deg=self.deg, **self._kwargs)
#         # self._coef = [np.polyfit(x=np.arange(0, X.shape[0]), y=X[:, i], deg=self.deg, **self._kwargs) for i in
#         #          range(X.shape[1])]
#         return self
#
#     def transform(self, X):
#         # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
#         #                 warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES)
#         self._trend = np.polyval(p=self._coef, x=np.arange(0, X.shape[0]))
#         X[self.var] = X[self.var] - self._trend
#         return X
#
#     def inverse_transform(self, X):
#         X[self.var] = X[self.var] + self._trend
#
#
# class DeTrendOp3(BaseEstimator, TransformerMixin):
#     """
#     name: the variable name you want to create
#     var: variable in with to compute the transformation
#     var_lower: variable in with to compute the transformation (lower "bound")
#     var_upper: variable in with to compute the transformation (upper "bound")
#     """
#
#     def __init__(self, name, var, var_lower, var_upper):
#         self.name = name
#         self.var = var
#         self.var_lower = var_lower
#         self.var_upper = var_upper
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X[self.name] = X[self.var] * (X[self.var_upper] - X[self.var_lower]) + X[self.var_lower]
#         return X
#
#
# class DeTrendOp2(BaseEstimator, TransformerMixin):
#     """
#     name: the variable name you want to create
#     var: variable in with to compute the transformation
#     var_lower: variable in with to compute the transformation (lower "bound")
#     var_upper: variable in with to compute the transformation (upper "bound")
#     """
#
#     def __init__(self, name, var, var_lower, var_upper):
#         self.name = name
#         self.var = var
#         self.var_lower = var_lower
#         self.var_upper = var_upper
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X[self.name] = (X[self.var] - X[self.var_lower]) / (X[self.var_upper] - X[self.var_lower])
#         return X
#
#
# class DeTrendOp1(BaseEstimator, TransformerMixin):
#     def __init__(self, name, var, by):
#         self.name = name
#         self.var = var
#         self.by = by
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X[self.name] = (X[self.var] - X[self.by])
#         return X
#
#
#
#
# # class Filter4Lag(BaseEstimator, TransformerMixin):
# #     def __init__(self, tol):
# #         self.tol = tol
# #
# #     def fit(self, X, y=None):
# #         return self
# #
# #     def transform(self, X, y=None):
# #         n_original = X.shape[0]
# #         X['tvalue'] = X.index
# #         X['delta'] = (X['tvalue'] - X['tvalue'].shift()).fillna(0)
# #         X['diff_delta'] = X['delta'].apply(lambda x: x / np.timedelta64(1, 'm')).astype('int64') % (24 * 60)
# #         X = X[X.diff_delta <= self.tol]
# #         X.drop(['tvalue', 'delta', 'diff_delta'], axis=1, inplace=True)
# #         n_final = X.shape[0]
# #         print("%f '%s of data lost with filter" % ((1 - n_final / n_original) * 100, '%'))
# #
# #         return X
#
#
# class DataFrameCompute(BaseEstimator, TransformerMixin):
#     def __init__(self, what="corr", transpose=False, fill_na=None):
#         self.what = what
#         # todo: convert to function
#         self.transpose = transpose
#         self.fill_na = fill_na
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         if self.transpose:
#             XX = eval("X.T." + self.what + "()")
#         else:
#             XX = eval("X." + self.what + "()")
#         if self.fill_na is not None:
#             return XX.fillna(self.fill_na)
#         else:
#             return XX
#
#
# class AddPolynomialFeatures(BaseEstimator, TransformerMixin):
#     """Generate polynomial and interaction features.
#
#     Generate a new feature matrix consisting of all polynomial combinations
#     of the features with degree less than or equal to the specified degree.
#     For example, if an input sample is two dimensional and of the form
#     [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
#
#     Parameters
#     ----------
#     degree : integer
#         The degree of the polynomial features. Default = 2.
#
#     interaction_only : boolean, default = False
#         If true, only interaction features are produced: features that are
#         products of at most ``degree`` *distinct* input features (so not
#         ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).
#
#     include_bias : boolean
#         If True (default), then include a bias column, the feature in which
#         all polynomial powers are zero (i.e. a column of ones - acts as an
#         intercept term in a linear model).
#
#     exclude_feat : list, default = None
#         list of features to exclude from the polynomial feature creation
#     """
#
#     def __init__(self, degree=2, interaction_only=False, include_bias=True, exclude_feat=None):
#         self.degree = degree
#         self.interaction_only = interaction_only
#         self.include_bias = include_bias
#         self.exclude_feat = exclude_feat
#         self._t = None
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         # remove object attributes
#         XX = X.select_dtypes(exclude=['object']).copy()
#         # remove extract_feat
#         XX = XX[XX.columns.difference([self.exclude_feat])]
#         XX_c = X.select_dtypes(include=['object'])
#         XX_c = pd.concat([XX_c, X[self.exclude_feat]], axis=1)
#         self._t = pp.PolynomialFeatures(degree=self.degree,
#                                         interaction_only=self.interaction_only,
#                                         include_bias=self.include_bias)
#         self._t.fit(XX)
#         XX = pd.DataFrame(self._t.transform(XX), columns=self._t.get_feature_names(XX.columns), index=X.index)
#         XX = pd.concat([XX, XX_c], axis=1)
#         return XX
#
#
# class TimeSeriesGroupBy(BaseEstimator, TransformerMixin):
#     def __init__(self, by, agg={'mean'}):
#         self.by = by
#         self.agg = agg
#         self.fitX_ = None
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         self.fitX_ = X.groupby(self.by).agg(self.agg)
#         self.fitX_.columns = ['__'.join(col) for col in self.fitX_.columns]
#         self.fitX_ = self.fitX_.T
#         return self.fitX_
#
#
# class TransformByColumnCategory(BaseEstimator, TransformerMixin):
#     """
#     cat column in which the categories are present
#     transformer
#     col_t columns in which the transformer will be applied
#     """
#
#     def __init__(self, cat, transformer=pp.StandardScaler(), col_t=None):
#         self.cat = cat
#         self.transformer = transformer
#         self._transformers = None
#         self._c = [col_t] if isinstance(col_t, str) else col_t
#         self.col_t = col_t
#
#     def fit(self, X, y=None):
#         # computes the number of unique categories cat has
#
#         if self._c is not None:
#             if self.cat not in self._c:
#                 self._c.append(self.cat)
#             X = X[self._c]
#         # nc = X[self.cat].unique().__len__()
#         c = X[self.cat].unique()
#         # initializes the nc transformers need
#         self._transformers = dict(zip(c, [clone(self.transformer) for i in c]))
#         for i in c:
#             self._transformers[i].fit(X[X[self.cat] == i].drop(self.cat, axis=1))
#
#         return self
#
#     def transform(self, X):
#         if self.col_t is not None:
#             XX = X[self._c].copy()
#         else:
#             XX = X.copy()
#         XXX = pd.DataFrame()
#         nnp_flag = 0
#         for i in self._transformers.keys():
#             res = self._transformers[i].transform(XX[XX[self.cat] == i].drop(self.cat,
#                                                                              axis=1))
#             if isinstance(res, np.ndarray):
#                 XX.loc[XX[self.cat] == i, self.col_t] = res
#             else:
#                 nnp_flag = 1
#                 XXX = pd.concat([res, XXX])
#
#         if nnp_flag == 1:
#             XX = XXX.copy()
#             del XXX
#             X = X.ix[XX.index]
#
#         if self.col_t is not None:
#             X[self.col_t] = XX[self.col_t]
#             return X
#         else:
#             return XX
#
#     def inverse_transform(self, X):
#         if self.col_t is not None:
#             XX = X[self._c].copy()
#         else:
#             XX = X.copy()
#         XXX = pd.DataFrame()
#         nnp_flag = 0
#         for i in self._transformers.keys():
#             res = self._transformers[i].inverse_transform(XX[XX[self.cat] == i].drop(self.cat,
#                                                                                      axis=1))
#             if isinstance(res, np.ndarray):
#                 XX.loc[XX[self.cat] == i, self.col_t] = res
#             else:
#                 nnp_flag = 1
#                 XXX = pd.concat([res, XXX])
#
#         if nnp_flag == 1:
#             XX = XXX.copy()
#             del XXX
#             X = X.ix[XX.index]
#
#         if self.col_t is not None:
#             X[self.col_t] = XX[self.col_t]
#             return X
#         else:
#             return XX
#
#
# class OutlierMAD(BaseEstimator, TransformerMixin):
#     def __init__(self, var, threshold=3):
#         self.threshold = threshold
#         self.var = var
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         XX = X[self.var]
#         if len(XX.shape) == 1:
#             XX = XX[:, None]
#         median = np.median(XX)
#         diff = np.sum((XX - median) ** 2, axis=-1)
#         diff = np.sqrt(diff)
#         d_median = np.median(diff)
#         if np.abs(d_median) < np.finfo(np.float32).eps:
#             d_median = 1.0
#         modified_z_score = 0.6745 * diff / d_median
#         print(np.median(diff))
#         out = modified_z_score > self.threshold
#         return X[~out]

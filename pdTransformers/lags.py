from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import warnings


class InsertLags(BaseEstimator, TransformerMixin):
    """
    Insert lags using shift method

    :param lags: : dict, dictionary with reference of the columns and number of lags for each column

    Example:
            from WWIR.pd_transformers.datasets import generate_ts
            dataset = generate_ts(n_samples=1000, n_features=2, n_targets=1,
                                    split_X_y=False, start_date='2016-01-03 00:00',
                                    freq='1H')
            from WWIR.pd_transformers.lags import InsertLags
            nlags = 3
            IL = InsertLags(lags={0: np.arange(1, nlags + 1), })

    """

    def __init__(self, lags):
        self.lags = lags

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.sort_index()
        if type(X) is pd.DataFrame:
            original_cols = X.columns.values.tolist()
            subs = [item for item in self.lags.keys()]
            original_cols = [i for i, item in enumerate(original_cols) if item in subs]

            new_dict = {}
            for predictor, all_lags in self.lags.items():
                if predictor not in X.columns:
                    warnings.warn(' ## Lags from \'{}\' were excluded, since the dataset wasn\'t loaded.'.format(
                        predictor))
                    continue
                new_dict[predictor] = X[predictor]
                for l in all_lags:
                    X['%s_lag%d' % (predictor, l)] = X[predictor].shift(l)
            return X
        elif type(X) is pd.Series:
            X = pd.concat([X.shift(i) for i in self.lags], axis=1)
            return X

    def inverse_transform(self, X, y=None):
        return X


class InsertAggByTime(BaseEstimator, TransformerMixin):
    """
    Inserts aggregated features (p.e. averages)

    :param agg: dict
        dictionary with variable name (column) as key, key-item as tuple (aggregator, by)
    :param timev: default='index'
        column translating the time.
    :param dropna: default=False,
        flag indicating if the NaN are to be dropped

    Example
        from WWIR.pd_transformers.datasets import generate_ts
        dataset = generate_ts(n_samples=1000, n_features=2, n_targets=1,
                              split_X_y=False, start_date='2016-01-03 00:00',
                              freq='1H')
        from WWIR.pd_transformers.ts_transformers import InsertAggByTimeLags as IATL
        c = IATL(agg_lags={'target': [('mean', '5min')]})
        c.fit_transform(dataset)
    """

    def __init__(self, agg, timev='index', dropna=False):

        self.agg = agg
        self.timev = timev
        self.dropna = dropna

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # @todo use re-sample function instead
        XX = X.copy()

        XX.reset_index(inplace=True)
        if self.timev == 'index':
            self.timev = X.index.name

        if hasattr(X, 'columns'):
            original_cols = X.columns.values.tolist()
            subs = [item for item in self.agg.keys()]
            original_cols = [i for i, item in enumerate(original_cols) if item in subs]
        else:
            original_cols = list(range(len(X[0, :])))

        for predictor, all_lags in self.agg.items():
            if predictor not in X.columns:
                warnings.warn(
                    ' ## ERROR! Lags from \'{}\' were excluded, since the dataset wasn\'t loaded.'.format(predictor))
                continue

            df = XX[[self.timev, predictor]]

            for tuple_ in all_lags:
                if tuple_.__len__() == 2:
                    agg, by = tuple_
                    ar = {}
                elif tuple_.__len__() == 3:
                    agg, by, ar = tuple_
                else:
                    print('parameters not well defined')

                XX['ts'] = XX[self.timev].values.astype('<M8[' + by + ']')

                df['ts'] = XX['ts']
                df_agg = eval("df.groupby(['ts'])." + agg + "(**" + ar.__str__() + ")")
                if isinstance(df_agg.index, pd.core.index.MultiIndex):
                    df_agg = df_agg.unstack(None)

                XX = XX.merge(df_agg, left_on='ts', right_index=True, how='left',
                              suffixes=('', '_%s_%s' % (agg.lower(), by)))
                if 'ts' in XX.columns:
                    XX.drop('ts', axis=1, inplace=True)
                if ('%s_%s_%s' % (self.timev, agg.lower(), by)) in XX.columns:
                    XX.drop('%s_%s_%s' % (self.timev, agg.lower(), by), axis=1, inplace=True)

        XX.set_index(self.timev, inplace=True)
        if self.dropna:
            XX.dropna(inplace=True)
        return XX

    def inverse_transform(self, X, y=None):
        pass

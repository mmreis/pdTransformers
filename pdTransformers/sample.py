# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.externals.joblib import Memory

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.fixes import bincount
import random


# class SampleAgglomerativeClustering(BaseEstimator, TransformerMixin):
#     """
#
#             :param n_clusters:
#             :param affinity:
#             :param memory:
#             :param connectivity:
#             :param compute_full_tree:
#             :param linkage:
#             :param pooling_func:
#             """
#
#     def __init__(self, n_samples, affinity='euclidean',
#                  memory=Memory(cachedir=None, verbose=0), connectivity=None,
#                  compute_full_tree='auto',
#                  linkage='ward', pooling_func=np.mean):
#         self.n_clusters = n_samples
#         self.affinity = affinity
#         self.memory = memory
#         self.connectivity = connectivity
#         self.compute_full_tree = compute_full_tree
#         self.linkage = linkage
#         self.pooling_func = pooling_func
#         self._model = []
#         self.weights = []
#         self.y_ = []
#         self.X_ = []
#
#     def fit(self, X, y=None):
#         self._model = AgglomerativeClustering(n_clusters=self.n_clusters,
#                                               affinity=self.affinity,
#                                               memory=self.memory,
#                                               connectivity=self.connectivity,
#                                               compute_full_tree=self.compute_full_tree,
#                                               linkage=self.linkage,
#                                               pooling_func=self.pooling_func)
#         self._model.fit(X=X)
#         df_ = pd.DataFrame(X)
#         col = df_.columns
#         df_['target'] = y
#         df_['labels_'] = self._model.labels_
#         df = df_.groupby('labels_').mean()
#         xx = df_.groupby('labels_').count()
#         df['weights'] = xx[xx.columns[0]]
#         del df_
#         self.weights = np.array(df['weights'])
#         self.y_ = np.array(df['target'])
#         self.X_ = np.array(df[col])
#         return self
#
#     def transform(self, X):
#         return self.X_, self.y_, self.weights


class StratifiedSample(BaseEstimator, TransformerMixin):
    """
    Method for Sampling, provides a sample with same distribution.

    :param var: str
        Variable used in the sampling decision process.
    :param sample_size: int or float (default=0.2)
        Sample size, can be a percentage of the total or a fixed number.
    :param nbins: int (default=10)
        If var is a continuous variable, a categorization is performed using nbins.
    :param random_state: int or RandomState
        Pseudo-random number generator state used for random sampling.
    """
    def __init__(self, var, sample_size=0.2, nbins=10, random_state=None):
        """


        """
        self.var = var
        self.sample_size = sample_size
        self.random_state = random_state
        self.nbins = nbins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # n = X.shape[0]

        # check if column is in X
        if self.var not in X.columns:
            ValueError("X does not contain variable {} ".format(self.var))

        # check if column is categorical, if not categorize
        classes, y_indices = np.unique(y, return_inverse=True)
        class_counts = bincount(y_indices)
        if np.min(class_counts) < 2:
            X['%s__cat' % self.var] = pd.cut(X[self.var], self.nbins).cat.codes
            var = '%s__cat' % self.var
        else:
            var = self.var

        split = StratifiedShuffleSplit(n_splits=1, train_size=self.sample_size, random_state=self.random_state)
        for train_index, test_index in split.split(X, X[var]):
            XX = X.iloc[train_index, :]
            # XX = X.loc[test_index]

        if '%s__cat' % self.var in X.columns:
            XX.drop('%s__cat' % self.var, axis=1, inplace=True)

        return XX


class BudgetSample(BaseEstimator, TransformerMixin):
    def __init__(self, var, sample_size=0.2, nbins=10, random_state=None):
        self.var = var
        self.sample_size = sample_size
        self.random_state = random_state
        self.nbins = nbins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n = X.shape[0]
        # check if column is in X
        if self.var not in X.columns:
            ValueError("X does not contain variable {} ".format(self.var))

        # check if column is categorical, if not categorize
        classes, y_indices = np.unique(y, return_inverse=True)
        class_counts = bincount(y_indices)
        if np.min(class_counts) < 2:
            X['%s__cat' % self.var] = pd.cut(X[self.var], self.nbins).cat.codes
            var = '%s__cat' % self.var
        else:
            var = self.var

        # compute budget per bin
        if isinstance(self.sample_size, int):
            budget_per_bin = int(self.sample_size / self.nbins)
        else:
            budget_per_bin = int(self.sample_size / self.nbins * X.shape[0])

        obs_by_cat = X.groupby(var).count().loc[:, self.var]

        keep_indexes = []
        for i in obs_by_cat.index:
            if obs_by_cat.iloc[i] <= budget_per_bin:
                keep_indexes += list(np.where(X[var] == i)[0])
            else:
                keep_indexes += random.sample(list(np.where(X[var] == i)[0]), budget_per_bin)

        return X.iloc[keep_indexes]



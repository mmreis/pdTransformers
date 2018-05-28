import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
import inspect


def generate_ts(n_samples=100, n_features=10, n_informative=10,
                start_date=None, freq='D', tz=None,
                n_targets=1, bias=0.0, effective_rank=None,
                tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
                random_state=None, split_X_y=True):
    """
    Function for generating synthetic multivariate time series.
    Returns a pandas object with index with date type.

    :param n_samples: int, optional (default=100)
        The number of samples.
    :param n_features: int, optional (default=10)
        The number of features.
    :param n_informative: int, optional (default=10)
        The number of informative features, i.e., the number of features used
        to build the linear model used to generate the output.
    :param start_date: string or datetime-like, default None
        Left bound for generating dates
    :param freq: string or DateOffset, default 'D' (calendar daily)
        Frequency strings can have multiples, e.g. '5H'
    :param tz: string or None
        Time zone name for returning localized DatetimeIndex, for example
    :param n_targets: int, optional (default=1)
        The number of regression targets, i.e., the dimension of the y output
        vector associated with a sample.
    :param bias:float, optional (default=0.0)
        The bias term in the underlying linear model.
    :param effective_rank:int or None, optional (default=None)
        if not None:
            The approximate number of singular vectors required to explain most
            of the input data by linear combinations. Using this kind of
            singular spectrum in the input allows the generator to reproduce
            the correlations often observed in practice.
        if None:
            The input set is well conditioned, centered and gaussian with
            unit variance.
    :param tail_strength: float between 0.0 and 1.0, optional (default=0.5)
        The relative importance of the fat noisy tail of the singular values
        profile if `effective_rank` is not None.
    :param noise: float, optional (default=0.0)
        The standard deviation of the gaussian noise applied to the output.
    :param shuffle: boolean, optional (default=True)
        Shuffle the samples and the features.
    :param coef:boolean, optional (default=False)
        If True, the coefficients of the underlying linear model are returned.
    :param random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :param split_X_y: boolean, optional (default=True)
        If True, the function returns two data-frames X and target y.
        If False, a pandas data-frame is returned where the last n_targets-columns
        are the targets.

    generate_ts(start_date='2016-01-01', n_samples=20, n_features=3)
    """
    d_index = pd.date_range(start=start_date, periods=n_samples, freq=freq, tz=tz)
    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                           n_targets=n_targets, bias=bias, effective_rank=effective_rank,
                           tail_strength=tail_strength, noise=noise, shuffle=shuffle, coef=coef,
                           random_state=random_state)
    X = pd.DataFrame(X, index=d_index, columns=['M' + str(i+1) for i in range(n_features)])
    cols_y = ['target'] if n_targets == 1 else ['target_%d' % i for i in range(n_targets)]
    y = pd.DataFrame(y, index=d_index, columns=cols_y)
    if split_X_y:
        return X, y
    else:
        return pd.concat([X, y], axis=1)


def generate_ts_with_nans(percentage=0.01, **kwargs):
    """
    Function for generating synthetic multivariate time series which contains NaN's
    Returns a pandas object with index with date type.
    :param percentage:
    :param kwargs: arguments for generate_ts
    """
    dataset = generate_ts(**kwargs)
    n, d = dataset.shape
    n_nans = int(dataset.size * percentage)
    ind = np.random.choice(dataset.size, size=(n_nans,))
    ind_ = np.zeros((n_nans, 2))
    ind_[:, 0] = ind / d
    ind_[:, 1] = ind % d
    ind_ = (ind_).astype(int)
    dat = dataset.values
    dat[ind_[:, 0], ind_[:, 1]] = np.nan

    return pd.DataFrame(dat, index=dataset.index, columns=dataset.columns)

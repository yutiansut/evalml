import numpy as np

from sklearn.model_selection import TimeSeriesSplit as SkTimeSeriesSplit
from sklearn.model_selection._split import BaseCrossValidator


class TimeSeriesSplit(BaseCrossValidator):

    def __init__(self, max_lag, gap, n_folds=3):
        self.max_lag = max_lag
        self.gap = gap
        self.n_folds = n_folds
        self._splitter = SkTimeSeriesSplit(n_splits=n_folds)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._splitter.n_splits

    def split(self, X, y=None, groups=None):
        max_index = X.shape[0]
        for train, test in self._splitter.split(X, y, groups):
            last_train = train[-1]
            last_test = test[-1]
            first_test = test[0]
            max_test_index = min(last_test + 1 + self.gap, max_index)
            new_train = np.concatenate([train, np.arange(last_train + 1, last_train + 1 + self.gap)])
            new_test = np.concatenate([
                np.arange(first_test - self.max_lag, first_test), test, np.arange(last_test + 1, max_test_index)
            ])
            yield new_train, new_test

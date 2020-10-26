import pandas as pd

from evalml.automl.data_splitters import TimeSeriesSplit


def test_time_series_split():
    X = pd.DataFrame({"features": range(1, 32)})
    y = pd.Series(range(1, 32))
    y.index = pd.date_range("2020-10-01", "2020-10-31")
    X.index = pd.date_range("2020-10-01", "2020-10-31")

    answer = [(pd.date_range("2020-10-01", "2020-10-12"), pd.date_range("2020-10-07", "2020-10-19")),
              (pd.date_range("2020-10-01", "2020-10-19"), pd.date_range("2020-10-14", "2020-10-26")),
              (pd.date_range("2020-10-01", "2020-10-26"), pd.date_range("2020-10-21", "2020-10-31"))]

    ts_split = TimeSeriesSplit(gap=2, max_lag=4)
    for i, (train, test) in enumerate(ts_split.split(X, y)):
        X_train = X.iloc[train]
        y_test = y.iloc[test]
        pd.testing.assert_index_equal(X_train.index, answer[i][0])
        pd.testing.assert_index_equal(y_test.index, answer[i][1])

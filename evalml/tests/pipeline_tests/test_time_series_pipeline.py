from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.pipelines import TimeSeriesRegressionPipeline
from evalml.pipelines.time_series import _add_time_series_parameters
from evalml.problem_types import ProblemTypes, TimeSeriesProblem


@pytest.fixture
def ts_data():
    X, y = pd.DataFrame({"features": range(101, 132)}), pd.Series(range(1, 32))
    y.index = pd.date_range("2020-10-01", "2020-10-31")
    X.index = pd.date_range("2020-10-01", "2020-10-31")
    return X, y


def test_add_time_series_parameters():
    ts = TimeSeriesProblem(gap=2, n_periods_to_predict=2, estimator_type="regression", date_column="date", max_lag=4)
    parameters = {"Lagged Feature Extractor": {}, "One Hot Encoder": {"top_n": 5}}
    answer = {"Lagged Feature Extractor": {"gap": 2, "n_periods_to_predict": 2,
                                           "estimator_type": ProblemTypes.REGRESSION,
                                           "date_column": "date",
                                           "series_id": None,
                                           "unit": "day",
                                           "max_lag": 4},
              "One Hot Encoder": {"top_n": 5}}
    assert _add_time_series_parameters(parameters, ts) == answer


@pytest.mark.parametrize("pipeline_class", [TimeSeriesRegressionPipeline])
@pytest.mark.parametrize("components", [["One Hot Encoder"],
                                        ["One Hot Encoder", "Lagged Feature Extractor"]])
def test_time_series_pipeline_init(pipeline_class, components):
    ts = TimeSeriesProblem(gap=3, n_periods_to_predict=2, estimator_type="regression", date_column="date", max_lag=5)

    class Pipeline(pipeline_class):
        component_graph = components + ["Random Forest Regressor"]

    pl = Pipeline({}, ts)
    if "Lagged Feature Extractor" not in components:
        assert "Lagged Feature Extractor" not in pl.parameters
    else:
        assert pl.parameters['Lagged Feature Extractor'] == {"gap": 3, "n_periods_to_predict": 2,
                                                             "estimator_type": ProblemTypes.REGRESSION,
                                                             "date_column": "date",
                                                             "series_id": None,
                                                             "unit": "day",
                                                             "max_lag": 5}


@pytest.mark.parametrize("include_lagged_features", [True, False])
@pytest.mark.parametrize("gap,max_lag", [(1, 2), (2, 2), (7, 3), (2, 4)])
@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor")])
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
def test_fit_drop_nans_before_estimator(mock_regressor_fit, pipeline_class,
                                        estimator_name, gap, max_lag, include_lagged_features, ts_data):

    X, y = ts_data

    ts = TimeSeriesProblem(gap=gap, n_periods_to_predict=2, estimator_type="regression", date_column="date",
                           max_lag=max_lag)

    if include_lagged_features:
        components = ["Lagged Feature Extractor", estimator_name]
        train_index = pd.date_range(f"2020-10-{1 + max_lag}", f"2020-10-{31-gap}")
        expected_target = np.arange(1 + gap + max_lag, 32)
        target_index = pd.date_range(f"2020-10-{1 + max_lag}", f"2020-10-{31 - gap}")
    else:
        components = [estimator_name]
        train_index = pd.date_range(f"2020-10-01", f"2020-10-{31-gap}")
        expected_target = np.arange(1 + gap, 32)
        target_index = pd.date_range(f"2020-10-01", f"2020-10-{31-gap}")

    class Pipeline(pipeline_class):
        component_graph = components

    pl = Pipeline({}, ts)
    pl.fit(X, y)
    df_passed_to_estimator, target_passed_to_estimator = mock_regressor_fit.call_args[0]
    assert not df_passed_to_estimator.isna().any(axis=1).any()
    assert not target_passed_to_estimator.isna().any()
    pd.testing.assert_index_equal(df_passed_to_estimator.index, train_index)
    pd.testing.assert_index_equal(target_passed_to_estimator.index, target_index)
    np.testing.assert_equal(target_passed_to_estimator.values, expected_target)


@pytest.mark.parametrize("include_lagged_features", [True, False])
@pytest.mark.parametrize("gap,max_lag", [(1, 2), (2, 2), (7, 3), (2, 4)])
@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor")])
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.RandomForestRegressor.predict")
def test_predict_pad_nans(mock_regressor_predict, mock_regressor_fit,
                          pipeline_class,
                          estimator_name, gap, max_lag, include_lagged_features, ts_data):

    X, y = ts_data
    ts = TimeSeriesProblem(gap=gap, n_periods_to_predict=2, estimator_type="regression", date_column="date",
                           max_lag=max_lag)

    def mock_predict(df):
        return pd.Series(range(200, 200 + df.shape[0]))

    mock_regressor_predict.side_effect = mock_predict

    if include_lagged_features:
        components = ["Lagged Feature Extractor", estimator_name]
    else:
        components = [estimator_name]

    class Pipeline(pipeline_class):
        component_graph = components

    pl = Pipeline({}, ts)
    pl.fit(X, y)
    preds = pl.predict(X, y)
    if include_lagged_features:
        assert np.isnan(preds.values[:max_lag]).all()
    else:
        assert not np.isnan(preds.values).any()


@pytest.mark.parametrize("include_lagged_features", [True, False])
@pytest.mark.parametrize("gap,max_lag", [(1, 2), (2, 2), (7, 3), (2, 4)])
@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor")])
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.RandomForestRegressor.predict")
@patch("evalml.pipelines.RegressionPipeline._score_all_objectives")
def test_score_drops_nans(mock_score, mock_regressor_predict, mock_regressor_fit,
                          pipeline_class,
                          estimator_name, gap, max_lag, include_lagged_features, ts_data):

    X, y = ts_data

    ts = TimeSeriesProblem(gap=gap, n_periods_to_predict=2, estimator_type="regression", date_column="date",
                           max_lag=max_lag)

    def mock_predict(df):
        return pd.Series(range(200, 200 + df.shape[0]))

    mock_regressor_predict.side_effect = mock_predict

    if include_lagged_features:
        components = ["Lagged Feature Extractor", estimator_name]
        expected_target = np.arange(1 + gap + max_lag, 32)
        target_index = pd.date_range(f"2020-10-{1 + max_lag}", f"2020-10-{31 - gap}")
    else:
        components = [estimator_name]
        expected_target = np.arange(1 + gap, 32)
        target_index = pd.date_range(f"2020-10-01", f"2020-10-{31-gap}")

    class Pipeline(pipeline_class):
        component_graph = components

    pl = Pipeline({}, ts)
    pl.fit(X, y)

    pl.score(X, y, objectives=[])
    _, target, preds = mock_score.call_args[0]
    assert not target.isna().any()
    assert not preds.isna().any()

    pd.testing.assert_index_equal(target.index, target_index)
    np.testing.assert_equal(target.values, expected_target)

import pytest

from evalml.pipelines import TimeSeriesRegressionPipeline
from evalml.pipelines.time_series import _add_time_series_parameters
from evalml.problem_types import ProblemTypes, TimeSeriesProblem


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

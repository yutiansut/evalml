from evalml.objectives import get_objective
from evalml.pipelines.regression_pipeline import RegressionPipeline
from evalml.problem_types import ProblemTypes


def pad_with_nans(*args):
    return args


def drop_nan(*args):
    return args


class TimeSeriesRegressionPipeline(RegressionPipeline):
    problem_type = ProblemTypes.TIME_SERIES_REGRESSION

    def __init__(self, parameters, random_state=0,
                 time_series_problem=None):
        super().__init__(parameters, random_state)
        self.time_series_problem = time_series_problem

    def fit(self, X, y):
        # This shift will introduce nans but PipelineBase._fit will remove them
        # Need to shift to account for the gap parameter
        y_shifted = y.shift(-self.time_series_problem.gap)
        super().fit(X, y_shifted)

    # Need to update the API to accept the target variable
    # So that we can eventually compute lags of the target variable
    def predict(self, X, y=None, objective=None):
        # Need to drop nans before feeding to the estimator
        features = self.compute_estimator_features(X, y)
        predictions = self.estimator.predict(features.dropnan())
        return pad_with_nans(predictions, self.time_series_problem)

    def score(self, X, y, objectives):
        # Override score to not change ObjectiveBase
        y_shifted = y.shift(-self.time_series_problem.gap)
        objectives = [get_objective(o, return_instance=True) for o in objectives]
        y_predicted = self.predict(X, y)
        y_shifted, y_predicted = drop_nan(y_shifted, y_predicted)
        return self._score_all_objectives(X, y_shifted,
                                          y_predicted,
                                          y_pred_proba=None,
                                          objectives=objectives)

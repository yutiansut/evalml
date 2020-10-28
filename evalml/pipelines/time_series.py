import copy

import pandas as pd

from evalml.objectives import get_objective
from evalml.pipelines.components.utils import handle_component_class
from evalml.pipelines.regression_pipeline import RegressionPipeline
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import any_values_are_nan, drop_nan, pad_with_nans


def _add_time_series_parameters(parameters, time_series_problem):
    new_params = copy.deepcopy(parameters)
    for class_name, class_params in new_params.items():
        component_class = handle_component_class(class_name)
        if hasattr(component_class, "needs_time_series_parameters"):
            class_params.update(time_series_problem.to_dict())
            new_params[class_name] = class_params

    return new_params


def _add_all_components_to_parameters(component_graph, parameters):
    new_params = copy.deepcopy(parameters)
    for component in component_graph:
        if component not in parameters:
            new_params[component] = {}
    return new_params


class TimeSeriesRegressionPipeline(RegressionPipeline):
    problem_type = ProblemTypes.TIME_SERIES_REGRESSION

    def __init__(self, parameters, time_series_problem, random_state=0):
        all_params = _add_all_components_to_parameters(self.component_graph, parameters)
        updated_params = _add_time_series_parameters(all_params, time_series_problem)
        super().__init__(updated_params, random_state)
        self.time_series_problem = time_series_problem

    def fit(self, X, y=None):
        if y is None:
            assert isinstance(X, pd.Series), "When modeling a single time series, the data must be a series."
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if y is None:
            X_t, _ = self._compute_features_during_fit(X, y=None)
            y = X.squeeze()
        else:
            y = pd.Series(y)
            X_t, _ = self._compute_features_during_fit(X, y)

        y_shifted = y.shift(-self.time_series_problem.gap)
        X_t, y_shifted = drop_nan(X_t, y_shifted)
        self.estimator.fit(X_t, y_shifted)
        return self

    def predict(self, X, y=None, objective=None):
        features = self.compute_estimator_features(X, y)
        predictions = self.estimator.predict(features.dropna(axis=0, how="any"))
        if any_values_are_nan(features):
            return pad_with_nans(predictions, self.time_series_problem.max_lag)
        else:
            return predictions

    def score(self, X, y, objectives):
        y_predicted = self.predict(X, y)
        if y is None:
            y = X.copy()
        y_shifted = y.shift(-self.time_series_problem.gap)
        objectives = [get_objective(o, return_instance=True) for o in objectives]
        y_shifted, y_predicted = drop_nan(y_shifted, y_predicted)
        return self._score_all_objectives(X, y_shifted,
                                          y_predicted,
                                          y_pred_proba=None,
                                          objectives=objectives)

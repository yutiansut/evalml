from .utils import handle_problem_types


class TimeSeriesProblem:
    """Parametrization of a time series problem."""
    def __init__(self, max_lag, gap, n_periods_to_predict,
                 estimator_type, date_column, series_id=None, unit="day"):
        self.max_lag = max_lag
        self.gap = gap
        self.n_periods_to_predict = n_periods_to_predict
        self.estimator_type = handle_problem_types(estimator_type)
        self.date_column = date_column
        self.series_id = series_id
        self.unit = unit

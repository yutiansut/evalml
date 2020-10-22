from evalml.pipelines.regression_pipeline import RegressionPipeline
from evalml.problem_types import ProblemTypes


class TimeSeriesRegressionPipeline(RegressionPipeline):
    problem_type = ProblemTypes.TIME_SERIES_REGRESSION

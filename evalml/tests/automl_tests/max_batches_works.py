from unittest.mock import patch
from evalml.automl import AutoMLSearch
from evalml.problem_types import ProblemTypes
import itertools
from sklearn import datasets


def test_max_batches_works(max_batches, use_ensembling, problem_type):
    X_y_binary = datasets.make_classification(n_samples=100, n_features=20,
                                              n_informative=2, n_redundant=2, random_state=0)

    X_y_regression = datasets.make_regression(n_samples=100, n_features=20,
                                              n_informative=3, random_state=0)

    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        automl = AutoMLSearch(problem_type="binary", max_iterations=None,
                              max_batches=max_batches, ensembling=use_ensembling)
    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        automl = AutoMLSearch(problem_type="regression", max_iterations=None,
                              max_batches=max_batches, ensembling=use_ensembling)

    automl.search(X, y, data_checks=None)
    # every nth batch a stacked ensemble will be trained
    ensemble_nth_batch = len(automl.allowed_pipelines) + 1

    if max_batches is None:
        n_results = 5
        max_batches = 1
        # _automl_algorithm will include all allowed_pipelines in the first batch even
        # if they are not searched over. That is why n_automl_pipelines does not equal
        # n_results when max_iterations and max_batches are None
        n_automl_pipelines = 1 + len(automl.allowed_pipelines)
        num_ensemble_batches = 0
    else:
        # automl algorithm does not know about the additional stacked ensemble pipelines
        num_ensemble_batches = (max_batches - 1) // ensemble_nth_batch if use_ensembling else 0
        # So that the test does not break when new estimator classes are added
        n_results = 1 + len(automl.allowed_pipelines) + (5 * (max_batches - 1 - num_ensemble_batches)) + num_ensemble_batches
        n_automl_pipelines = n_results
    assert automl._automl_algorithm.batch_number == max_batches
    assert automl._automl_algorithm.pipeline_number + 1 == n_automl_pipelines
    assert len(automl.results["pipeline_results"]) == n_results
    if num_ensemble_batches == 0:
        assert automl.rankings.shape[0] == min(1 + len(automl.allowed_pipelines), n_results)  # add one for baseline
    else:
        assert automl.rankings.shape[0] == min(2 + len(automl.allowed_pipelines), n_results)  # add two for baseline and stacked ensemble
    assert automl.full_rankings.shape[0] == n_results


if __name__ == "__main__":

    for max_batches, use_ensembling, problem_type in itertools.product([None, 1, 5, 8, 9, 10, 12, 20], [False, True],
                                                                       [ProblemTypes.BINARY, ProblemTypes.REGRESSION]):
        with patch('evalml.pipelines.RegressionPipeline.score', return_value={"R2": 0.8}), \
            patch('evalml.pipelines.RegressionPipeline.fit'), \
            patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8}), \
            patch('evalml.pipelines.BinaryClassificationPipeline.fit'):

            test_max_batches_works(max_batches, use_ensembling, problem_type)

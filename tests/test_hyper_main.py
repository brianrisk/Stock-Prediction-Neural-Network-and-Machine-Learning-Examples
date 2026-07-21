import unittest
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

if importlib.util.find_spec('pytorch_lightning') is None:
    raise unittest.SkipTest('pytorch_lightning is not installed')

from hyperparameter_tuning.hyper_main import (
    get_hyperparameter_combinations, run_model_for_hyperparameters,
    evaluate_hyperparameters, store_results
)
from hyperparameter_tuning.config import hyperparameter_values


class TestHyperMain(unittest.TestCase):

    @patch('hyperparameter_tuning.hyper_main.EXPLORE_ALL_COMBINATIONS', True)
    def test_get_hyperparameter_combinations_all(self):
        combinations = get_hyperparameter_combinations()
        expected_count = 1
        for values in hyperparameter_values.values():
            expected_count *= len(values)
        self.assertEqual(len(combinations), expected_count)

    @patch('hyperparameter_tuning.hyper_main.EXPLORE_ALL_COMBINATIONS', False)
    @patch('hyperparameter_tuning.hyper_main.NUMBER_OF_COMBINATIONS_TO_TRY', 5)
    def test_get_hyperparameter_combinations_subset(self):
        combinations = get_hyperparameter_combinations()
        self.assertEqual(len(combinations), 5)

    @patch('hyperparameter_tuning.hyper_main.get_ffnn')
    @patch('hyperparameter_tuning.hyper_main.get_data')
    @patch('hyperparameter_tuning.hyper_main.DataLoader')
    @patch('hyperparameter_tuning.hyper_main.L.Trainer')
    def test_run_model_for_hyperparameters(
        self, mock_trainer, mock_dataloader, mock_get_data, mock_get_ffnn
    ):
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.__call__.return_value = MagicMock()
        mock_model.__call__.return_value.numpy.return_value = [0.6, 0.4]
        mock_get_ffnn.return_value = mock_model
        mock_get_data.return_value = (MagicMock(), [[0.0], [1.0]], [1, 0], 1)

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        params = {key: 1 for key in hyperparameter_values.keys()}
        p_value = run_model_for_hyperparameters(params, mock_get_ffnn)
        self.assertIsNotNone(p_value)

    @patch('hyperparameter_tuning.hyper_main.run_model_for_hyperparameters', return_value=0.05)
    @patch('hyperparameter_tuning.hyper_main.get_ffnn')
    @patch('hyperparameter_tuning.hyper_main.get_data')
    def test_evaluate_hyperparameters(self, mock_get_data, mock_get_ffnn, mock_run_model):
        mock_get_data.return_value = (MagicMock(), [[0.0]], [1], 1)
        params = {key: 1 for key in hyperparameter_values.keys()}
        result = evaluate_hyperparameters((1, params.values()))
        self.assertIsNotNone(result)
        self.assertEqual(result[1], 0.05)

    def test_store_results(self):
        params = {key: 1 for key in hyperparameter_values.keys()}
        results = [{**params, 'p_value': 0.05, 'execution_time': 10, 'seed': 42}]
        errors = [{**params, 'error': 'Some error', 'seed': 42}]
        with TemporaryDirectory() as directory:
            with patch('hyperparameter_tuning.hyper_main.RESULTS_DIR', Path(directory)):
                store_results(results, errors)
            self.assertTrue((Path(directory) / 'hyperparameter_results.csv').exists())
            self.assertTrue((Path(directory) / 'errors.csv').exists())


if __name__ == '__main__':
    unittest.main()

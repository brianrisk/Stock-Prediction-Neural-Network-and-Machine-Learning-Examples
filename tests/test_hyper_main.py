import unittest
from unittest.mock import patch, MagicMock
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
    @patch('hyperparameter_tuning.hyper_main.DataLoader')
    @patch('hyperparameter_tuning.hyper_main.L.Trainer')
    def test_run_model_for_hyperparameters(self, mock_trainer, mock_dataloader, mock_get_ffnn):
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.__call__.return_value = MagicMock()
        mock_model.__call__.return_value.numpy.return_value = [0.6, 0.4]
        mock_get_ffnn.return_value = mock_model

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        params = {key: 1 for key in hyperparameter_values.keys()}
        p_value = run_model_for_hyperparameters(params, mock_get_ffnn)
        self.assertIsNotNone(p_value)

    @patch('hyperparameter_tuning.hyper_main.run_model_for_hyperparameters', return_value=0.05)
    @patch('hyperparameter_tuning.hyper_main.get_ffnn')
    def test_evaluate_hyperparameters(self, mock_get_ffnn, mock_run_model):
        params = {key: 1 for key in hyperparameter_values.keys()}
        result = evaluate_hyperparameters((1, params.values()))
        self.assertIsNotNone(result)
        self.assertEqual(result[1], 0.05)

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('csv.DictWriter')
    def test_store_results(self, mock_dict_writer, mock_open):
        results = [{'param1': 1, 'p_value': 0.05, 'execution_time': 10}]
        errors = [{'param1': 1, 'error': 'Some error'}]
        store_results(results, errors)
        self.assertTrue(mock_open.called)
        self.assertTrue(mock_dict_writer.called)


if __name__ == '__main__':
    unittest.main()

import unittest

from evaluation import backtest_top_scores, classification_metrics, probability_metrics


class TestEvaluation(unittest.TestCase):
    def test_classification_metrics(self):
        metrics = classification_metrics(tp=8, fp=2, fn=2, tn=8)
        self.assertAlmostEqual(metrics['f1'], 0.8)
        self.assertAlmostEqual(metrics['balanced_accuracy'], 0.8)
        self.assertEqual(metrics['coverage'], 0.5)

    def test_probability_metrics(self):
        metrics = probability_metrics([0, 1], [0.1, 0.9])
        self.assertEqual(metrics['average_precision'], 1.0)
        self.assertAlmostEqual(metrics['brier_score'], 0.01)

    def test_backtest_accounts_for_cost(self):
        result = backtest_top_scores(
            scores=[0.9, 0.8, 0.1, 0.0],
            realized_returns=[0.10, 0.06, -0.02, 0.0],
            fraction=0.5,
            transaction_cost=0.01,
        )
        self.assertAlmostEqual(result['net_return'], 0.07)
        self.assertAlmostEqual(result['excess_return'], 0.035)


if __name__ == '__main__':
    unittest.main()

import unittest

from common import calculate_precision_p_value


class TestCalculatePrecisionPValue(unittest.TestCase):
    def test_returns_significant_value_for_enriched_predictions(self):
        self.assertLess(calculate_precision_p_value(20, 2, 5, 30), 0.05)

    def test_returns_one_when_no_positive_predictions_exist(self):
        self.assertEqual(calculate_precision_p_value(0, 0, 5, 30), 1.0)

    def test_returns_one_when_precision_does_not_beat_base_rate(self):
        self.assertEqual(calculate_precision_p_value(1, 9, 9, 1), 1.0)


if __name__ == '__main__':
    unittest.main()

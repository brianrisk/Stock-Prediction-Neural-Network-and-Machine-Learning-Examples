import unittest

import pandas as pd

from validation import chronological_split


class TestChronologicalSplit(unittest.TestCase):
    def test_preserves_order_and_embargo(self):
        data = pd.DataFrame({'row': range(10)})
        train, validation = chronological_split(data, validation_fraction=0.3, gap=1)
        self.assertEqual(train['row'].tolist(), list(range(7)))
        self.assertEqual(validation['row'].tolist(), [8, 9])

    def test_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            chronological_split(pd.DataFrame({'row': range(10)}), 1.0)


if __name__ == '__main__':
    unittest.main()

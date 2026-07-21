import tempfile
import unittest
from pathlib import Path

from data_utils import load_labeled_data, load_latest_data


class TestDataUtils(unittest.TestCase):
    def test_loads_valid_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'train.csv').write_text('1.0,0\n2.0,1\n')
            (path / 'latest.csv').write_text('ticker,feature\nABC,1.0\n')
            self.assertEqual(load_labeled_data('train.csv', path).shape, (2, 2))
            self.assertEqual(load_latest_data(path).iloc[0, 0], 'ABC')

    def test_rejects_non_binary_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'train.csv').write_text('1.0,2\n')
            with self.assertRaises(ValueError):
                load_labeled_data('train.csv', path)


if __name__ == '__main__':
    unittest.main()

import random
import unittest

import numpy as np

from reproducibility import seed_everything


class TestReproducibility(unittest.TestCase):
    def test_reseeding_repeats_random_values(self):
        seed_everything(7)
        first = (random.random(), np.random.random())
        seed_everything(7)
        self.assertEqual(first, (random.random(), np.random.random()))


if __name__ == '__main__':
    unittest.main()

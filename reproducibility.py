import random
import sys

import numpy as np

DEFAULT_SEED = 42


def seed_everything(seed=DEFAULT_SEED):
    """Seed Python, NumPy, and any already-imported ML frameworks."""
    random.seed(seed)
    np.random.seed(seed)

    if 'torch' in sys.modules:
        torch = sys.modules['torch']
        torch.manual_seed(seed)
    if 'tensorflow' in sys.modules:
        tensorflow = sys.modules['tensorflow']
        tensorflow.random.set_seed(seed)
    return seed

from pathlib import Path

import pandas as pd

from common import DATA_DIR


def load_labeled_data(filename, data_dir=DATA_DIR):
    """Load feature columns followed by one binary label column."""
    frame = pd.read_csv(Path(data_dir) / filename, header=None)
    if frame.shape[1] < 2:
        raise ValueError(f'{filename} must contain features and a label column')
    labels = set(frame.iloc[:, -1].dropna().unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f'{filename} labels must contain only 0 and 1')
    return frame


def load_latest_data(data_dir=DATA_DIR):
    """Load ticker identifiers followed by feature columns."""
    frame = pd.read_csv(Path(data_dir) / 'latest.csv')
    if frame.shape[1] < 2:
        raise ValueError('latest.csv must contain a ticker and feature columns')
    return frame


def split_features_and_labels(frame):
    return frame.iloc[:, :-1], frame.iloc[:, -1]

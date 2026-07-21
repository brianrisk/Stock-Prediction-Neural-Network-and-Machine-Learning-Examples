import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from data_utils import load_labeled_data
from hyperparameter_tuning.config import EMBARGO_ROWS, VALIDATION_FRACTION
from validation import chronological_split


def load_data():
    all_train_data = load_labeled_data('train.csv')
    train_data, validation_data = chronological_split(
        all_train_data,
        validation_fraction=VALIDATION_FRACTION,
        gap=EMBARGO_ROWS,
    )
    X = train_data.iloc[:, :-1].values
    Y = train_data.iloc[:, -1].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    train_dataset = TensorDataset(torch.tensor(X_scaled), torch.tensor(Y))

    X_validation = validation_data.iloc[:, :-1].values
    Y_validation = validation_data.iloc[:, -1].values
    X_validation_scaled = scaler.transform(X_validation)

    input_feature_size = X_scaled.shape[1]

    return train_dataset, X_validation_scaled, Y_validation, input_feature_size

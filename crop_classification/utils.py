import math
import torch
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def read_data(balanced_test=True):
    df = pd.read_csv("./data/WinnipegDataset.txt")
    y = df["label"] - 1
    X = df.drop(["label"], axis=1)[
        [f"f{i + j}" for j in [1, 50, 99, 137] for i in range(3)]
    ]
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    print("Counts per class before undersampling:")
    print(y.value_counts())

    _, X_te, _, y_te = train_test_split(X, y, test_size=0.01, random_state=42)
    X = X.drop(X_te.index, axis=0)
    y = y.drop(y_te.index, axis=0)

    undersampler = RandomUnderSampler(random_state=42)
    X, y = undersampler.fit_resample(X, y)
    print("Counts per class after undersampling:")
    print(y.value_counts())

    X_tr, X_te_balanced, y_tr, y_te_balanced = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if balanced_test:
        X_te = X_te_balanced
        y_te = y_te_balanced

    print("Counts per class in the test dataset:")
    print(y_te.value_counts())

    return (
        X_tr.values,
        X_te.values,
        y_tr.values,
        y_te.values,
    )


class TorchDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


def accuracy(y, y_hat):
    match = y - y_hat
    match[match != 0] = -1
    match += 1

    acc = match.sum() / match.shape[0]

    return acc

import math
import torch
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def read_data(undersampling=True):
    df = pd.read_csv("./data/WinnipegDataset.txt")
    Y = df["label"] - 1
    X = df.drop(["label"], axis=1)[
        [f"f{i + j}" for j in [1, 50, 99, 137] for i in range(3)]
    ]
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    print(Y.value_counts())

    if undersampling:
        undersampler = RandomUnderSampler(random_state=42)
        X, Y = undersampler.fit_resample(X, Y)
        print(Y.value_counts())

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_tr.values, X_te.values, Y_tr.values, Y_te.values


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

import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix


def read_data():
    df = pd.read_csv("./data/WinnipegDataset.txt")
    y = df["label"] - 1
    X = df.drop(["label"], axis=1)[
        [f"f{i + j}" for j in [1, 99, 50, 137] for i in range(3)]
    ]
    print("Counts per class before undersampling:")
    print(y.value_counts())

    _, X_te, _, y_te = train_test_split(X, y, test_size=0.005, random_state=42)
    X = X.drop(X_te.index, axis=0)
    y = y.drop(y_te.index, axis=0)

    undersampler = RandomUnderSampler(random_state=42)
    X, y = undersampler.fit_resample(X, y)
    X_tr, X_balanced, y_tr, y_balanced = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Counts per class in the training dataset:")
    print(y_tr.value_counts())
    print("Counts per class in the imbalanced test dataset:")
    print(y_te.value_counts())
    print("Counts per class in the balanced test datset:")
    print(y_balanced.value_counts())

    # min-max normalization
    X_te = (X_te - X_tr.min(0)) / (X_tr.max(0) - X_tr.min(0))
    X_balanced = (X_balanced - X_tr.min(0)) / (X_tr.max(0) - X_tr.min(0))
    X_tr = (X_tr - X_tr.min(0)) / (X_tr.max(0) - X_tr.min(0))

    return (
        X_tr.values,
        X_te.values,
        X_balanced.values,
        y_tr.values,
        y_te.values,
        y_balanced.values,
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


def print_metrics(y_pred, y_te):
    acc = accuracy_score(y_pred, y_te)
    p, r, f, _ = precision_recall_fscore_support(y_pred, y_te)

    print(f"Test Accuracy: {acc}")

    print(pd.DataFrame({"Precision": p, "Recall": r, "F1": f}))

    cf_matrix = confusion_matrix(y_te, y_pred)
    sns.set(font_scale=0.5)
    sns.heatmap(cf_matrix, annot=True, fmt="g")
    plt.show()

import torch
from torch.utils.data import Dataset
import numpy as np


class NFLPropsDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

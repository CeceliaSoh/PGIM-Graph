import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    """
    Dataset class for loading synthetic time series data from CSV.
    Creates sliding windows for training.
    """
    def __init__(self, csv_path, window_size=50, split='train', train_ratio=0.8, include_lag=True):
        self.csv_path = csv_path
        self.window_size = window_size
        self.include_lag = include_lag
        
        # Load data: [y, x1, x2, y_lag1]
        df = pd.read_csv(csv_path)
        
        # Ground truth is always the first column
        self.y = df.iloc[:, 0:1].values.astype(np.float32)
        
        # Features: x1, x2 are columns 1 and 2
        # y_lag1 is column 3
        if include_lag:
            self.features = df.iloc[:, 1:].values.astype(np.float32)
        else:
            # Exclude y_lag1 (column 3)
            self.features = df.iloc[:, 1:3].values.astype(np.float32)
        
        # Split into train and test
        n_total = len(df)
        n_train = int(n_total * train_ratio)
        
        if split == 'train':
            self.y = self.y[:n_train]
            self.features = self.features[:n_train]
        else:
            self.y = self.y[n_train:]
            self.features = self.features[n_train:]
            
    def __len__(self):
        if len(self.y) <= self.window_size:
            return 0
        return len(self.y) - self.window_size

    def __getitem__(self, idx):
        # Window of features: [window_size, feat_dim]
        # Window of targets: [window_size, 1]
        x_window = self.features[idx : idx + self.window_size]
        y_window = self.y[idx : idx + self.window_size]
        
        return torch.from_numpy(x_window), torch.from_numpy(y_window)

def get_synthetic_dataloader(csv_path, window_size=50, split='train', batch_size=32, shuffle=True, include_lag=True):
    """
    Utility to get a DataLoader for the synthetic dataset.
    """
    dataset = SyntheticDataset(csv_path, window_size, split, include_lag=include_lag)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle if split == 'train' else False,
        drop_last=False
    )
    return dataloader

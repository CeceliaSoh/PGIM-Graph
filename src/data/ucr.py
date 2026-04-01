import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class UCRDataset(Dataset):
    """
    Dataset class for loading UCR Time Series Archive data.
    Ensures data is returned in multivariate format [Seq_Len, Channels].
    """
    def __init__(self, root_dir, dataset_name, split='TRAIN'):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.split = split.upper()
        
        file_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_{self.split}")
        if not os.path.exists(file_path):
            # Try alternate path if not found (sometimes there is an extra level)
            file_path = os.path.join(root_dir, f"{dataset_name}_{self.split}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Could not find dataset file for {dataset_name} {split}")

        # Load data using numpy (UCR files are usually comma or space separated)
        try:
            # First attempt with comma separator
            data = np.loadtxt(file_path, delimiter=',')
        except ValueError:
            # Fallback to whitespace separator
            data = np.loadtxt(file_path)

        # First column is label, the rest is time series
        self.labels = data[:, 0].astype(np.int64)
        self.ts_data = data[:, 1:].astype(np.float32)

        # Normalize labels to start from 0 if necessary
        unique_labels = np.unique(self.labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        self.labels = np.array([label_map[l] for l in self.labels])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Time series data: [Seq_Len] -> [Seq_Len, 1] (Multivariate format)
        x = self.ts_data[idx]
        x = x[:, np.newaxis]
        
        y = self.labels[idx]
        
        return torch.from_numpy(x), torch.tensor(y)

def get_ucr_dataloader(root_dir, dataset_name, split='TRAIN', batch_size=32, shuffle=True):
    """
    Utility to get a DataLoader for a specific UCR dataset.
    """
    dataset = UCRDataset(root_dir, dataset_name, split)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle if split.upper() == 'TRAIN' else False,
        drop_last=False
    )
    return dataloader

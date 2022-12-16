import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate


def replace_labels(arr):
    arr[arr == 2] = 0
    arr = np.eye(2)[arr]
    return arr


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


class data_2c(Dataset):
    def __init__(self, csv, img_size=(32, 32, 32)):
        super().__init__()
        df = pd.read_csv(csv)
        self.paths = df.PATH.values
        self.labels = replace_labels(df.LABEL.values)
        self.img_size = img_size

    def __len__(self,):
        return len(self.paths)

    def __getitem__(self, idx):
        vol = np.load(self.paths[idx])
        vol = normalize(vol)[None, ...]
        vol = torch.tensor(vol, dtype=torch.float)
        vol = interpolate(vol, size=self.img_size)[0]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return vol, label

class aug_2c(Dataset):
    def __init__(self, csv, img_size=(32, 32, 32), aug_prob=0.5, noise_factor=0.1):
        super().__init__()
        df = pd.read_csv(csv)
        self.paths = df.PATH.values
        self.labels = replace_labels(df.LABEL.values)
        self.img_size = img_size
        self.aug_prob = aug_prob
        self.noise_factor = noise_factor

    def __len__(self,):
        return len(self.paths)

    def __getitem__(self, idx):
        vol = np.load(self.paths[idx])
        vol = normalize(vol)[None, ...]
        vol = torch.tensor(vol, dtype=torch.float)
        vol = interpolate(vol, size=self.img_size)[0]
        # with probability of self.aug_prob add noise to vol
        if np.random.rand() < self.aug_prob:
            vol = vol + torch.randn_like(vol) * self.noise_factor
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return vol, label



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import time
    dl = DataLoader(data_2c("data/test.csv"), 8)
    now = time.time()
    x, y = next(iter(dl))
    print(f"Time for one batch: {time.time()- now}")
    print(x.shape)
    print(y)
    breakpoint()

import torch
from torch.utils.data import Dataset

class data_2c(Dataset):
    def __init__(self, csv):
        super().__init__()
        self.csv = csv

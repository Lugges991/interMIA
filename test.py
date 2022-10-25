import os
import torch
import wandb
import numpy as np
import torch.optim as optim
import torchmetrics as tm
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.functional import interpolate

from interMIA.models import TwoCVGG
from interMIA.dataloader import data_2c


torch.manual_seed(42)

cfg = {"BATCH_SIZE": 30,
       "EPOCHS": 100,
       "LR": 0.1,
       "img_size": (32, 32, 32),
       "VAL_AFTER": 3,
       "MODEL_DIR": "./models",
       "MODEL_NAME": "TwoCVGG",
       }


def prepare_subs(df):
    subs = np.unique(df.SUB_ID.values)

    all_subs = []

    for sub in subs:
        sub_df = df[df.SUB_ID == sub]
        all_subs.append(sub_df)

    return all_subs

class TestData(Dataset):
    def __init__(self, df):
        super().__init__()
        self.paths = df.PATH.values
        self.labels = df.LABEL.values
        self.img_size=(32, 32, 32)

    def __len__(self, ):
        return len(self.paths)

    def __getitem__(self, idx):
        vol = np.load(self.paths[idx])[None, ...]
        vol = torch.tensor(vol, dtype=torch.float)
        vol = interpolate(vol, size=self.img_size)[0]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return vol, label




def test():
    test_data = pd.read_csv("data/sites/ABIDEII-KKI_1/test.csv")


    # model definition
    model = TwoCVGG().cuda()

    test_data = prepare_subs(test_data)

    model.eval()

    for sub in test_data:
        dat = TestData(sub)
        dl = DataLoader(dat, batch_size=cfg["BATCH_SIZE"], shuffle=False)

        label = sub.LABEL.iloc[0]

        preds = []
        for x,y in dl:
            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            breakpoint()
            preds.append(*pred.argmax().tolist().detach().cpu())

        vote = np.argmax(np.bincount(np.array(preds)))


if __name__ == "__main__":
    test()

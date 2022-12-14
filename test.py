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
from torch.nn.functional import interpolate, softmax

# from interMIA.models import TwoCVGG
# from interMIA.models import TwoCC3D as Model
from interMIA.models import ViT3D
from interMIA.dataloader import data_2c


torch.manual_seed(42)

cfg = {"BATCH_SIZE": 16,
       "EPOCHS": 10,
       "LR": 1e-5,
       "img_size": (32, 32, 32),
       "VAL_AFTER": 2,
       "MODEL_DIR": "./models/",
       "MODEL_NAME": ViT3D(patch_size=16, heads=16, depth=24, dim=1024, mlp_dim=4096, dropout=0.5, emb_dropout=0.3),
       "loss": nn.CrossEntropyLoss(),
       "INFO": "normalize",
       "SITE": "WHOLE",
       "WEIGHT_DECAY": 0.1,
       "MOMENTUM": 0.9,
       }


RUN_NAME = ""


def prepare_subs(df):
    subs = np.unique(df.SUB_ID.values)

    all_subs = []

    for sub in subs:
        sub_df = df[df.SUB_ID == sub]
        all_subs.append(sub_df)

    return all_subs


def replace_labels(arr):
    arr[arr == 2] = 0
    arr = np.eye(2)[arr]
    return arr


class TestData(Dataset):
    def __init__(self, df):
        super().__init__()
        self.paths = df.PATH.values
        self.labels = replace_labels(df.LABEL.values)
        self.img_size = (32, 32, 32)

    def __len__(self, ):
        return len(self.paths)

    def __getitem__(self, idx):
        vol = np.load(self.paths[idx])[None, ...]
        vol = torch.tensor(vol, dtype=torch.float)
        vol = interpolate(vol, size=self.img_size)[0]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return vol, label


def test():
    test_data = pd.read_csv("data/test.csv")

    # model definition
    model = cfg["MODEL_NAME"].cuda()
    # model = TwoCVGG().cuda()
    # model.load_state_dict(torch.load("/mnt/DATA/models/brain-biomarker-sitev0-generous-planet-8/best_model.pth")["state_dict"])
    model.load_state_dict(torch.load("/home/lmahler/code/interMIA/models/brain-biomarker-whole-trafo-v0_rare-dawn-5/model_epoch_0.pth")["state_dict"])

    test_data = prepare_subs(test_data)

    model.eval()

    correct = 0

    for sub in test_data:
        dat = TestData(sub)
        dl = DataLoader(dat, batch_size=cfg["BATCH_SIZE"], shuffle=False)

        label = sub.LABEL.iloc[0]

        pred_probs = []
        lab_0 = 0
        lab_1 = 0
        for x, y in dl:
            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            pred = softmax(pred)
            # append probabilities to array
            pred_probs.append(pred.detach().cpu())

            # transform probabilities to label
            p_label = (pred.clone().detach() > 0.5) * 1
            lab_0 += torch.sum(p_label[:, 0]).item()
            lab_1 += torch.sum(p_label[:, 1]).item()

        # majority vote over all samples
        if lab_0 > lab_1:
            vote = 0
        else:
            vote = 1

        if int(vote) == int(label):
            correct += 1
        print(
            f"Prediction for subject {sub.SUB_ID.iloc[0]} with label {label} was: {vote}")

    print(80 * "*")
    print(f"Accuracy: {correct / len(test_data)}")


if __name__ == "__main__":
    test()

import os
import sys
import torch
import wandb
import signal
import shutil
import torch.optim as optim
import torchmetrics as tm
from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from interMIA.models import TwoCVGG
from interMIA.models import TwoCC3D
from interMIA.models import DM
from interMIA.models import ResNet, ViT3D
from interMIA.dataloader import data_2c
from interMIA.utils.file_utils import check_or_make_dir, copy_models
from interMIA.utils import EarlyStopping


torch.manual_seed(42)



cfg = {"BATCH_SIZE": 16,
       "EPOCHS": 1,
       "LR": 1e-5,
       "img_size": (32, 32, 32),
       "VAL_AFTER": 2,
       "MODEL_DIR": "./models/",
       "MODEL_NAME": ViT3D(patch_size=16, heads=16, depth=24, dim=1024, mlp_dim=4096),
       "loss": nn.CrossEntropyLoss(),
       "INFO": "normalize",
       "SITE": "WHOLE",
       "WEIGHT_DECAY": 0.1,
       "MOMENTUM": 0.9,
       }

RUN_NAME = ""


def train():
    train_data = data_2c("data/train.csv")

    train_loader = DataLoader(
        train_data, batch_size=cfg["BATCH_SIZE"], shuffle=True)

    # model definition
    model = cfg["MODEL_NAME"].cuda()
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    #optimizer = optim.SGD(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"], momentum=cfg["MOMENTUM"])

    # loss
    criterion = nn.CrossEntropyLoss().cuda()


    # Early Stopping


    # metrics

    for epoch in range(cfg["EPOCHS"]):
        epoch_loss = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            model.train()
            cnt = 0
            for x, y in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inp = x.cuda()
                lab = y.cuda()

                pred = model(inp)

                loss = criterion(pred, lab)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=f"{loss.item():10.3f}")

                cnt += 1

                if cnt >10:
                    break

            torch.save({"epoch": epoch, "state_dict": model.state_dict(
            ), "optimizer": optimizer.state_dict()}, "data/test_tranny.pth")


    model2 = cfg["MODEL_NAME"].cuda()
    sd = torch.load("data/test_tranny.pth")["state_dict"]
    model2.load_state_dict(sd)



if __name__ == "__main__":
    #signal.signal(signal.SIGINT, copy_models)
    train()
    # signal.pause()

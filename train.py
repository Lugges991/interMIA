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
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

from interMIA.models import TwoCVGG
from interMIA.models import TwoCC3D
from interMIA.models import DM
from interMIA.models import ResNet, ViT3D
from interMIA.dataloader import data_2c, aug_2c
from interMIA.utils.file_utils import check_or_make_dir, copy_models
from interMIA.utils import EarlyStopping


torch.manual_seed(42)



cfg = {"BATCH_SIZE": 16,
       "EPOCHS": 10,
       "LR": 8e-4,
       "img_size": (32, 32, 32),
       "VAL_AFTER": 2,
       "MODEL_DIR": "./models/",
       "MODEL_NAME": ViT3D(heads=12, depth=12, dim=1024, mlp_dim=3072),
       "loss": nn.CrossEntropyLoss(),
       "INFO": "normalize",
       "SITE": "WHOLE",
       "WEIGHT_DECAY": 0.1,
       "MOMENTUM": 0.9,
       }

RUN_NAME = ""


def train():
    train_data = aug_2c("data/train.csv")
    val_data = data_2c("data/val.csv")

    train_loader = DataLoader(
        train_data, batch_size=cfg["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(
        val_data, batch_size=cfg["BATCH_SIZE"], shuffle=True)

    # model definition
    model = cfg["MODEL_NAME"].cuda()
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    #optimizer = optim.SGD(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"], momentum=cfg["MOMENTUM"])

    # scheduler
    scheduler = LinearLR(optimizer)

    # loss
    criterion = nn.CrossEntropyLoss().cuda()


    # Early Stopping
    early_stopper = EarlyStopping(patience=3, min_delta=10)


    # metrics
    accuracy = tm.Accuracy().cuda()
    precision = tm.Precision().cuda()
    recall = tm.Recall().cuda()
    f1_score = tm.F1Score().cuda()

    project_name = "brain-biomarker-whole-trafo-v0"
    run = wandb.init(project=project_name, group="kyb", config=cfg)
    model_dir = cfg["MODEL_DIR"] + project_name + "_" + run.name
    check_or_make_dir(model_dir)

    best_acc = 0.

    for epoch in range(cfg["EPOCHS"]):
        epoch_loss = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            model.train()
            for x, y in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inp = x.cuda()
                lab = y.cuda()

                pred = model(inp)

                loss = criterion(pred, lab)

                wandb.log({"BCELoss": loss})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=f"{loss.item():10.3f}")

            wandb.log({"Epoch Loss": epoch_loss})
            scheduler.step()


        if epoch % cfg["VAL_AFTER"] == 0:
            print(80*"+")
            print("Running Evaluation")
            print(80*"+")
            model.eval()

            val_loss = 0.

            for ii, (xx, yy) in enumerate(tqdm(val_loader)):
                inp = xx.cuda()
                lab = yy.cuda()

                with torch.no_grad():
                    pred = model(inp)

                val_loss += nn.functional.cross_entropy(pred, lab).item()

                lab = lab.type(torch.int)

                acc = accuracy(pred, lab)
                prec = precision(pred, lab)
                rec = recall(pred, lab)
                f1 = f1_score(pred, lab)




            acc = accuracy.compute()
            prec = precision.compute()
            rec = recall.compute()
            f1 = f1_score.compute()

            wandb.log({"Accuracy": acc, "Precision": prec,
                       "Recall": rec, "F1 Score": f1, "Epoch Loss": epoch_loss})

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save({"epoch": epoch, "state_dict": model.state_dict(
                ), "optimizer": optimizer.state_dict()}, os.path.join(model_dir, "best_model.pth"))

            accuracy.reset()
            precision.reset()
            recall.reset()
            f1_score.reset()
            torch.save({"epoch": epoch, "state_dict": model.state_dict(
            ), "optimizer": optimizer.state_dict()}, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

            # early stopping here
            if early_stopper.early_stop(val_loss):
                print(80*"+")
                print(f"Early Stopping at epoch {epoch}")
                torch.save({"epoch": epoch, "state_dict": model.state_dict(
                ), "optimizer": optimizer.state_dict()}, os.path.join(model_dir, "early_stopping.pth"))

    #copy_models(None, None)


if __name__ == "__main__":
    #signal.signal(signal.SIGINT, copy_models)
    train()
    # signal.pause()

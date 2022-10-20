import os
import torch
import wandb
import torch.optim as optim
import torchmetrics as tm
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from interMIA.models import TwoCC3D
from interMIA.dataloader import data_2c


torch.manual_seed(42)

cfg = {"BATCH_SIZE": 16,
       "EPOCHS": 100,
       "LR": 0.001,
       "img_size": (32, 32, 32),
       "VAL_AFTER": 2,
       "MODEL_DIR": "./models"
       }


def train():
    train_data = data_2c("data/train.csv")
    val_data = data_2c("data/val.csv")

    train_loader = DataLoader(
        train_data, batch_size=cfg["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(
        val_data, batch_size=cfg["BATCH_SIZE"], shuffle=True)

    # model definition
    model = TwoCC3D().cuda()

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg["LR"])

    # loss
    criterion = nn.BCELoss().cuda()

    # metrics
    accuracy = tm.Accuracy().cuda()
    precision = tm.Precision().cuda()
    recall = tm.Recall().cuda()
    f1_score = tm.F1Score().cuda()
    roc_score = tm.ROC().cuda()

    wandb.init(project="brain-biomarker-v0", group="kyb", config=cfg)

    best_acc = 0.

    for epoch in range(cfg["EPOCHS"]):
        epoch_loss = 0

        model.train()
        for i, (x, y) in enumerate(tqdm(train_loader)):
            inp = x.cuda()
            lab = y.cuda()

            pred = model(inp)

            loss = criterion(pred, lab)

            wandb.log({"BCELoss": loss})

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % cfg["VAL_AFTER"] == 0:
            print(80*"+")
            print("Running Evaluation")
            print(80*"+")
            model.eval()

            for ii, (xx, yy) in enumerate(tqdm(val_loader)):
                inp = xx.cuda()
                lab = yy.cuda()

                with torch.no_grad():
                    pred = model(inp)

                acc = accuracy(pred, lab)
                prec = precision(pred, lab)
                rec = recall(pred, lab)
                f1 = f1_score(pred, lab)
                roc = roc_score(pred, lab)

            acc = acc.compute()
            prec = prec.compute()
            rec = rec.compute()
            f1 = f1.compute()
            roc = roc.compute()

            wandb.log({"Accuracy": acc, "Precision": prec,
                      "Recall": rec, "F1 Score": f1, "ROC": roc})

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save({"epoch": epoch, "state_dict": model.state_dict(
                ), "optimizer": optimizer.state_dict()}, os.path.join("./models/", "best_model.pth"))

            accuracy.reset()
            precision.reset()
            recall.reset()
            f1_score.reset()
            roc_score.reset()
            torch.save({"epoch": epoch, "state_dict": model.state_dict(
            ), "optimizer": optimizer.state_dict()}, os.path.join("./models/", f"model_epoch_{epoch}.pth"))

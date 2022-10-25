import os
import torch
import wandb
import torch.optim as optim
import torchmetrics as tm
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

# from interMIA.models import TwoCVGG
from interMIA.models import TwoCC3D
from interMIA.dataloader import data_2c


torch.manual_seed(42)

cfg = {"BATCH_SIZE": 32,
       "EPOCHS": 100,
       "LR": 0.1,
       "img_size": (32, 32, 32),
       "VAL_AFTER": 3,
       "MODEL_DIR": "./models",
       "MODEL_NAME": "TwoCC3D",
       }


def train():
    train_data = data_2c("data/sites/ABIDEII-KKI_1/train.csv")
    val_data = data_2c("data/sites/ABIDEII-KKI_1/val.csv")

    train_loader = DataLoader(
        train_data, batch_size=cfg["BATCH_SIZE"], shuffle=False)
    val_loader = DataLoader(
        val_data, batch_size=cfg["BATCH_SIZE"], shuffle=False)

    # model definition
    model = TwoCC3D().cuda()

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg["LR"])
    # optimizer = optim.SGD(model.parameters(), lr=cfg["LR"], weight_decay=0.01)

    # loss
    criterion = nn.BCEWithLogitsLoss().cuda()

    # metrics
    accuracy = tm.Accuracy().cuda()
    precision = tm.Precision().cuda()
    recall = tm.Recall().cuda()
    f1_score = tm.F1Score().cuda()

    wandb.init(project="brain-biomarker-site-v0", group="kyb", config=cfg)

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
                tepoch.set_postfix(loss=loss.item())

            wandb.log({"Epoch Loss": epoch_loss})

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
                ), "optimizer": optimizer.state_dict()}, os.path.join("./models/", "best_model.pth"))

            accuracy.reset()
            precision.reset()
            recall.reset()
            f1_score.reset()
            torch.save({"epoch": epoch, "state_dict": model.state_dict(
            ), "optimizer": optimizer.state_dict()}, os.path.join("./models/", f"model_epoch_{epoch}.pth"))


if __name__ == "__main__":
    train()

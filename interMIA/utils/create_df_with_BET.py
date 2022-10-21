import pandas as pd
from pathlib import Path


df = pd.read_csv("/mnt/DATA/datasets/ABIDEII/fsl_subs.csv")
bet_paths = Path("/mnt/DATA/datasets/ABIDEII").rglob("bet.nii.gz")
SITE = "ABIDEII-KKI_1"

bet_dl = []
cols = [*df.columns.values, "BET"]

for bet in bet_paths:
    sub_id = int(bet.parts[-4])
    row = df[df.SUB_ID == sub_id]
    if len(row) == 1:
        temp = dict(zip(cols, [*row.values[0], bet]))
        bet_dl.append(temp)

bet_df = pd.DataFrame(bet_dl)
bet_df.to_csv("./data/fsl_bet.csv", index=False)

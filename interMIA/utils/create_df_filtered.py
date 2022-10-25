import pandas as pd
from pathlib import Path


df = pd.read_csv("/mnt/DATA/datasets/ABIDEII/fsl_subs.csv")
bet_paths = Path("/mnt/DATA/datasets/preprocessed/site-ABIDEII/raw/ABIDEII-KKI_1").rglob("filtered_func_data.nii.gz")
SITE = "ABIDEII-KKI_1"

bet_dl = []
cols = [*df.columns.values, "FILTER"]

for bet in bet_paths:
    sub_id = int(bet.parts[-2].strip(".feat"))
    row = df[df.SUB_ID == sub_id]
    if len(row) == 1:
        temp = dict(zip(cols, [*row.values[0], bet]))
        bet_dl.append(temp)

bet_df = pd.DataFrame(bet_dl)
bet_df.to_csv("./data/fsl_filtered.csv", index=False)

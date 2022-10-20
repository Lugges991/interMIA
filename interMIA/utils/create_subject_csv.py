import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split


np.random.seed(seed=42)


def main(fsl_dir, csv_subjects, train_csv, test_csv, val_csv):
    prep_paths = Path(fsl_dir).glob("*.npy")
    df = pd.read_csv(csv_subjects)
    
    paths = []
    labels = []
    sub_ids = []

    for p in prep_paths:
        sub_id = re.sub("\D", "", p.name.split("_")[0])
        try:
            row = df[df.SUB_ID == int(sub_id)].iloc[0]
            label = row.DX_GROUP
            paths.append(p)
            labels.append(label)
            sub_ids.append(sub_id)
        except Exception as e:
            print(sub_id)


    all_df = pd.DataFrame(columns=["PATH", "LABEL", "SUB_ID"])
    all_df["PATH"] = paths
    all_df["LABEL"] = labels
    all_df["SUB_ID"] = sub_ids

    test_subs = np.random.choice(np.unique(sub_ids), int(len(np.unique(sub_ids))/10))


    test_df = all_df[all_df["SUB_ID"].isin(test_subs)]
    train_df = all_df[~all_df["SUB_ID"].isin(test_subs)]

    train_df, val_df = train_test_split(train_df, test_size=0.2)

    test_df.to_csv(test_csv)
    train_df.to_csv(train_csv)
    val_df.to_csv(val_csv)




if __name__ == "__main__":
    fsl_dir = "/mnt/DATA/datasets/preprocessed/2Cprep_ABIDEII"
    main(fsl_dir, "/mnt/DATA/datasets/ABIDEII/fsl_subs.csv",train_csv="data/train.csv", test_csv="data/test.csv", val_csv="data/val.csv")

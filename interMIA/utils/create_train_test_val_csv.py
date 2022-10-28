import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split


np.random.seed(seed=42)

def put_paths_in_df(df, prep_paths):

    paths = []
    labels = []
    sub_ids = []

    for p in prep_paths:
        sub_id = int(re.sub("\D", "", p.name.split("_")[0]))
        try:
            row = df[df.SUB_ID == sub_id].iloc[0]
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

    return all_df




def main(npy_dir, csv_subjects, train_csv, test_csv, val_csv):
    prep_paths = Path(npy_dir).glob("*.npy")
    df = pd.read_csv(csv_subjects)

    all_df = put_paths_in_df(df, prep_paths=prep_paths)
    
    test_TC= np.random.choice(df[df.DX_GROUP == 2].SUB_ID.values, 4)
    test_ASD = np.random.choice(df[df.DX_GROUP == 1].SUB_ID.values, 4)

    test_df = all_df[all_df["SUB_ID"].isin([*test_TC, *test_ASD])]
    train_df = all_df[~(all_df["SUB_ID"].isin([*test_TC, *test_ASD]))]

    train_df, val_df = train_test_split(train_df, test_size=0.2)

    test_df.to_csv(test_csv)
    train_df.to_csv(train_csv)
    val_df.to_csv(val_csv)


if __name__ == "__main__":
    npy_dir = "/mnt/DATA/datasets/preprocessed/site-ABIDEII/2Cprep/ABIDEII-GU_1"
    main(npy_dir, "data/fsl_filtered_ABIDEII-GU_1.csv",
         train_csv="data/sites/ABIDEII-GU_1/train.csv", test_csv="data/sites/ABIDEII-GU_1/test.csv", val_csv="data/sites/ABIDEII-GU_1/val.csv")

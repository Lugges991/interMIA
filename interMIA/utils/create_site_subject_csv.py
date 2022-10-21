import os
import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split


np.random.seed(seed=42)

fsl_dir = "/mnt/DATA/datasets/preprocessed/2Cprep_ABIDEII"
csv_subjects = "/mnt/DATA/datasets/ABIDEII/fsl_subs.csv"

out_dir = "./data/sites"


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split(site_df):
    subs = np.unique(site_df.SUB_ID.values)
    test_subs = np.random.choice(subs, len(subs) // 10)

    test_df = site_df[site_df.SUB_ID.isin(test_subs)]
    train_df = site_df[~site_df.SUB_ID.isin(test_subs)]
    return train_df, test_df


def main():

    prep_paths = Path(fsl_dir).glob("*.npy")
    df = pd.read_csv(csv_subjects)

    if not os.path.isfile("./data/sub_site_dx_path.csv"):

        all_sub = []

        for path in prep_paths:
            sub_id = re.sub("\D", "", path.name.split("_")[0])
            row = df[df.SUB_ID == int(sub_id)].iloc[0]
            temp = {"SUB_ID": sub_id, "PATH": path,
                    "LABEL": row.DX_GROUP, "SITE_ID": row.SITE_ID}
            all_sub.append(temp)

        new_df = pd.DataFrame(all_sub)
        new_df.to_csv("./data/sub_site_dx_path.csv", index=False)

    else:
        new_df = pd.read_csv("./data/sub_site_dx_path.csv")

    sites = np.unique(df.SITE_ID.values)

    for site in sites:
        site_df = new_df[new_df.SITE_ID == site]
        train, test = split(site_df)
        print(f"{site} with len {len(site_df)}")

        if len(site_df) > 0:
            train, val = train_test_split(train, test_size=0.2)
            path = out_dir + f"/{site}"
            make_dir(path)

            train.to_csv(path+f"/{site}_train.csv", index=False)
            val.to_csv(path+f"/{site}_val.csv", index=False)
            test.to_csv(path+f"/{site}_test.csv", index=False)


main()

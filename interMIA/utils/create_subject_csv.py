import pandas as pd
import re
from pathlib import Path


def main(fsl_dir, csv_subjects, out_csv):
    prep_paths = Path(fsl_dir).glob("*.npy")
    df = pd.read_csv(csv_subjects)
    
    paths = []
    labels = []

    for p in prep_paths:
        sub_id = re.sub("\D", "", p.name.split("_")[0])
        try:
            row = df[df.SUB_ID == int(sub_id)].iloc[0]
            label = row.DX_GROUP
            paths.append(p)
            labels.append(label)
        except Exception as e:
            print(sub_id)


    out_df = pd.DataFrame(columns=["PATH", "LABEL"])
    out_df["PATH"] = paths
    out_df["LABEL"] = labels
    out_df.to_csv(out_csv)


if __name__ == "__main__":
    fsl_dir = "/mnt/DATA/datasets/preprocessed/2Cprep_ABIDEII"
    main(fsl_dir, "/mnt/DATA/datasets/ABIDEII/fsl_subs.csv",out_csv="data/subjects_x_y.csv")

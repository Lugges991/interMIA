import pandas as pd
from pathlib import Path
import os
import time
from multiprocessing import Pool


out_dir = "/mnt/DATA/datasets/preprocessed/site-ABIDEII/raw/ABIDEII-KKI_1/"


def search_replace_template(fmri, out, tr, vol, bet, design, base_template="./data/design_template.fsf"):

    with open(base_template, "r") as f:
        file = f.read()
        f.close()

    fmri = f'set feat_files(1) "{fmri}"'
    out = f'set fmri(outputdir) "{out}"'
    tr = f'set fmri(tr) {tr:.6f}'
    vol = f'set fmri(npts) {int(vol)}'
    bet = f'set highres_files(1) "{bet}"'

    file = file.replace("FINPUT", fmri)
    file = file.replace("ODIR", out)
    file = file.replace("TEER", tr)
    file = file.replace("TV", vol)
    file = file.replace("EXT", bet)

    with open(design, "w") as f:
        f.write(file)


def run_feat(row):

    start = time.time()

    # get paths from df
    fmri = row.PATH
    anat = row.ANAT_PATH
    tr = row.TR
    vol = row.VOL
    sub_id = row.SUB_ID
    bet = row.BET

    # construnct output path/mnt/DATA/datasets/preprocessed/ABIDEII
    out_path = out_dir + str(sub_id)
    design = f"/tmp/design_{sub_id}.fsf"

    print(f"Starting FEAT for subject {sub_id}...")
    # change template
    search_replace_template(fmri, out_path, tr, vol, bet, design)

    # run feat
    stream = os.popen(f"feat {design}")
    print(stream.read())
    end = time.time()
    print(f"Finished subject {sub_id} in {end-start} seconds.")
    print(80 * "*")


if __name__ == "__main__":
    df = pd.read_csv("./data/fsl_bet.csv")
    df = df[df.SITE_ID == "ABIDEII-KKI_1"]
    with Pool(10) as p:
        print(p.map(run_feat, [row for i, row in df.iterrows()]))

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os


in_dir = "/mnt/DATA/datasets/preprocessed/site-ABIDEII/raw/ABIDEII-GU_1"
out_name = "reg/func_reg.nii.gz"

mat_name = "reg/example_func2standard.mat"
std_name = "reg/standard.nii.gz"

path_list = Path(in_dir).glob("*")

subs = pd.read_csv("data/fsl_filtered_ABIDEII-GU_1.csv")


for path in tqdm(path_list):

    sub_id = int(path.parts[-1].strip(".feat"))

    in_vol = subs[subs.SUB_ID == sub_id]
    in_path = in_vol.FILTER.iloc[0]
    stream = os.popen(f"flirt -in {in_path} -ref {str(path.joinpath(std_name))} -applyxfm -init {str(path.joinpath(mat_name))} -out {str(path.joinpath(out_name))}")
    print(stream.read())

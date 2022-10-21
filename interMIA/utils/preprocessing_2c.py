"""Script that takes FSL preprocessed data, performs temporal slicing, averages over temporal slice, calculates std for temporal slice and stores as 2 channel 3D volume"""
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

def avg_std(slice):
    avg = np.mean(slice, axis=-1)
    std = np.mean(slice, axis=-1)
    return (avg, std)


def preprocess_volume(nifti_vol, temporal_win=3, skip_first=3):
    data = nifti_vol.get_fdata()[:, :, :, skip_first:]

    vol_arr = []
    for i in range(data.shape[-1]-temporal_win):
        vol_arr.append(np.stack(avg_std(data[:, :, :, i:temporal_win+i])))
    

    return vol_arr

def save_files(out_dir, sub_id, arr):
    for i, a in enumerate(arr):
        f_name = out_dir.joinpath(f"{sub_id}_{i}.npy")
        np.save(f_name, a)


def main(in_dir, out_dir):
    out_dir = Path(out_dir)
    func_list = Path(in_dir).rglob("filtered_func_data.nii.gz")
    for f_path in tqdm(func_list):
        sub_id = f_path.parts[-2].strip(".feat")
        vol = nib.load(f_path)
        arr = preprocess_volume(vol)
        save_files(out_dir, sub_id, arr)




if __name__ == "__main__":
    # example_vol = "/home/lmahler/code/interMIA/data/filtered_func_data.nii.gz"
    in_dir = "/mnt/DATA/datasets/preprocessed/site-ABIDEII/raw/ABIDEII-KKI_1"
    out_dir = "/mnt/DATA/datasets/preprocessed/site-ABIDEII/2Cprep/ABIDEII-KKI_1"
    main(in_dir, out_dir)

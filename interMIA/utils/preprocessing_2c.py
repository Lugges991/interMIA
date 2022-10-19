"""Script that takes FSL preprocessed data, performs temporal slicing, averages over temporal slice, calculates std for temporal slice and stores as 2 channel 3D volume"""
import pandas as pd
from pathlib import Path
import nibabel as nib

import nibabel as nib
import numpy as np

saliency_map = nib.load("data/saliency_standard_prep.nii.gz").get_fdata()
# cereb_mask_nii = nib.load("/home/lmahler/builds/fsl/data/atlases/MNI/MNI-maxprob-thr0-2mm.nii.gz")
cereb_mask_nii = nib.load("data/shen_2mm.nii.gz")
cereb_mask = cereb_mask_nii.get_fdata()


sal_mask = np.zeros(saliency_map.shape)
for i in np.unique(cereb_mask):
    # if saliency map is not zero in this region, set the mask to 1
    if np.any(saliency_map[cereb_mask == i] != 0):
        sal_mask[cereb_mask == i] = i


brain_mask = nib.load(
    "/home/lmahler/builds/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz").get_fdata()
sal_mask[brain_mask == 0] = 0

new_img = nib.Nifti1Image(sal_mask, cereb_mask_nii.affine)
nib.save(new_img, "data/saliency_standard_prep_mask.nii.gz")

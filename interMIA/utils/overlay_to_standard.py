import nibabel as nib
import numpy as np
from OrthoSlicer3D import OrthoSlicer3D as OS3D


def overlay_to_standard(overlay, standard, standard_mask=None, overlay_mask=None):
    """
    Overlay an overlay image to a standard image, using the standard image's affine matrix.
    """
    if standard_mask is None:
        standard_mask = np.ones(standard.shape)
    if overlay_mask is None:
        overlay_mask = np.ones(overlay.shape)
    # load the overlay image
    overlay_img = nib.Nifti1Image(overlay, np.eye(4))
    overlay_mask_img = nib.Nifti1Image(overlay_mask, np.eye(4))
    # load the standard image
    standard_img = nib.Nifti1Image(standard, np.eye(4))
    standard_mask_img = nib.Nifti1Image(standard_mask, np.eye(4))
    # resample the overlay image to the standard image's affine matrix
    overlay_to_standard = nib.Nifti1Image(
        nib.resampling.resample_from_to(
            overlay_img, standard_img, order=1, mode="constant"
        )[0],
        standard_img.affine,
    )
    overlay_mask_to_standard = nib.Nifti1Image(
        nib.resampling.resample_from_to(
            overlay_mask_img, standard_mask_img, order=1, mode="constant"
        )[0],
        standard_mask_img.affine,
    )
    return overlay_to_standard, overlay_mask_to_standard


if __name__ == "__main__":
    img = nib.load("/home/lmahler/builds/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz")
    breakpoint()
    OS3D(img.get_fdata(), cmap="gray").show()



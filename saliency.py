import torch
import numpy as np
from interMIA.models import ResNet
from interMIA.vis import compare_fm_differences
from OrthoSlicer3D import OrthoSlicer3D as OS3D
from captum.attr import Saliency
import nibabel as nib
from torch.nn.functional import interpolate
from interMIA.dataloader import normalize
from matplotlib import pyplot as plt


try:
    attribution = torch.load("data/saliency_TC123.pt")
    img_pt = attribution[0, 0, :, :, :]
except FileNotFoundError:
    # model
    model = ResNet()

    inp = np.load(
        "/mnt/DATA/datasets/preprocessed/2Cprep_ABIDEII/28813_97.npy")
    inp = interpolate(torch.from_numpy(inp)[None, ...].float(), size=(32, 32, 32))

    # attribution = sal.attribute(inp, target=1, n_steps=500)
    # img_no_pt = attribution.detach().numpy()[0, 0, :, :, :]
    # ortho = OS3D(img_no_pt, cmap="coolwarm").show()

    sd = torch.load(
        "models/brain-biomarker-site-v0_pious-galaxy-47/best_model.pth")["state_dict"]
    model.load_state_dict(sd)

    sal = Saliency(model)

    # if attributions.npy exists, load it
    # else, compute it
    attribution = sal.attribute(inp, target=0)
    # save tensor to disk
    torch.save(attribution, "data/saliency_TC.pt")
    img_pt = attribution.detach().numpy()[0, 0, :, :, :]
# ortho = OS3D(img_pt, cmap="coolwarm").show()

standard = nib.load(
    "/home/lmahler/builds/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz")
struct_seg = nib.load(
    "/home/lmahler/builds/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz")

img_rescale = normalize(interpolate(attribution, size=standard.shape)[
                        0, 0].detach().numpy())

# set values of img_rescale to 0 where struct_seg is 0
img_rescale[struct_seg.get_fdata() == 0] = 0

# set values of img_rescale to 0 where smaller than Mean
# img_rescale[img_rescale < np.median(img_rescale)] = 0


# only keep values in img_rescale that are half standard deviation above or below the mean
# img_rescale[img_rescale < np.mean(img_rescale) - np.std(img_rescale)] = 0
# img_rescale[img_rescale > np.mean(img_rescale) + np.std(img_rescale)] = 0

# only keep values in img_rescale that are two standard deviations above or below the mean
# img_rescale[img_rescale < np.mean(img_rescale) - 2 * np.std(img_rescale)] = 0
# img_rescale[img_rescale > np.mean(img_rescale) + 2 * np.std(img_rescale)] = 0

# only keep values in img_recale that are greater than 0.1
# img_rescale[img_rescale < 0.4] = 0
# img_rescale[img_rescale > 0.4] = 1


# gaussian blur img_rescale
# from scipy.ndimage import gaussian_filter
# img_rescale = gaussian_filter(img_rescale, sigma=0.5)

# uniform blur img_rescale
# from scipy.ndimage import uniform_filter
# img_rescale = uniform_filter(img_rescale, size=3)
# 

OS3D(img_rescale, cmap="coolwarm").show()

# plot and save image
plt.imshow(img_rescale[:, 15, :], cmap="coolwarm")
plt.savefig("data/vis/saliency_map.png")
plt.show()

# create new nifti image from img_rescale
# new_img = nib.Nifti1Image(img_rescale, standard.affine, standard.header)

# save new image
# nib.save(new_img, "data/saliency_standard_prep_TC.nii.gz")
# compare_fm_differences(img_no_pt, img_pt)

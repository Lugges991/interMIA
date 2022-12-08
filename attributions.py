import torch
import numpy as np
from interMIA.models import ResNet
from interMIA.vis import compare_fm_differences
from OrthoSlicer3D import OrthoSlicer3D as OS3D
from captum.attr import IntegratedGradients
import nibabel as nib
from torch.nn.functional import interpolate
from interMIA.dataloader import normalize


try:
    attribution = torch.load("data/attributions.pt")
    img_pt = attribution[0, 0, :, :, :]
except FileNotFoundError:
    # model
    model = ResNet()


    ig = IntegratedGradients(model)
    inp = torch.randn(1, 2, 32, 32, 32, requires_grad=True)

    # attribution = ig.attribute(inp, target=1, n_steps=500)
    # img_no_pt = attribution.detach().numpy()[0, 0, :, :, :]
    # ortho = OS3D(img_no_pt, cmap="coolwarm").show()

    sd = torch.load("models/brain-biomarker-site-v0_pious-galaxy-47/best_model.pth")["state_dict"]
    model.load_state_dict(sd)

    ig = IntegratedGradients(model)

    # if attributions.npy exists, load it
    # else, compute it
    attribution = ig.attribute(inp, target=0, n_steps=500)
    # save tensor to disk
    torch.save(attribution, "data/attributions.pt")
    img_pt = attribution.detach().numpy()[0, 0, :, :, :]
# ortho = OS3D(img_pt, cmap="coolwarm").show()

standard = nib.load("/home/lmahler/builds/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz")
struct_seg = nib.load("/home/lmahler/builds/fsl/data/standard/MNI152_T1_2mm_strucseg.nii.gz")

img_rescale = normalize(interpolate(attribution, size=standard.shape)[0, 0].detach().numpy())
# absolute values of img_rescale
img_rescale = np.abs(img_rescale)

# set values of img_rescale to 0 where struct_seg is 0
img_rescale[struct_seg.get_fdata() == 0] = 0

# set values of img_rescale to 0 where smaller than Mean
# img_rescale[img_rescale < np.median(img_rescale)] = 0

# only keep values in img_rescale that are half standard deviation above or below the mean

# only keep values in img_rescale that are half standard deviation above or below the mean
img_rescale[img_rescale < np.mean(img_rescale) - 0.6*np.std(img_rescale)] = 0
img_rescale[img_rescale > np.mean(img_rescale) + 0.6*np.std(img_rescale)] = 0

# only keep values in img_rescale that are significantly different from the mean 


# # get singular values of img_rescale
# u, s, vh = np.linalg.svd(img_rescale, full_matrices=False)
# 
# # plot singular values
# import matplotlib.pyplot as plt
# plt.plot(s)
# plt.show()
# 

# create new nifti image from img_rescale
new_img = nib.Nifti1Image(img_rescale, standard.affine, standard.header)

# save new image
nib.save(new_img, "data/attributions_standard_prep.nii.gz")
# compare_fm_differences(img_no_pt, img_pt)



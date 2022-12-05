import torch
import numpy as np
from interMIA.models import ResNet
from interMIA.vis import compare_fm_differences
from OrthoSlicer3D import OrthoSlicer3D as OS3D
from captum.attr import Saliency
# model
model = ResNet()

# saliency on randomly initialized model
sal = Saliency(model)
inp = torch.randn(1, 2, 32, 32, 32, requires_grad=True)

attr = sal.attribute(inp, target=0)
img_no_pt = attr.detach().numpy()[0, 0, :, :, :]

ortho = OS3D(img_no_pt, cmap="coolwarm").show()


# saliency on trained model
sd = torch.load("models/brain-biomarker-site-v0_pious-galaxy-47/best_model.pth")["state_dict"]
model.load_state_dict(sd)

sal = Saliency(model)

attr = sal.attribute(inp, target=0)
img_pt = attr.detach().numpy()[0, 0, :, :, :]

ortho = OS3D(img_no_pt, cmap="coolwarm").show()
compare_fm_differences(img_no_pt, img_pt)



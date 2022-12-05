import torch
import numpy as np
from interMIA.models import ResNet
from interMIA.vis import compare_fm_differences
from OrthoSlicer3D import OrthoSlicer3D as OS3D
from captum.attr import IntegratedGradients
# model
model = ResNet()


ig = IntegratedGradients(model)
inp = torch.randn(1, 2, 32, 32, 32, requires_grad=True)

attribution = ig.attribute(inp, target=1, n_steps=500)
img_no_pt = attribution.detach().numpy()[0, 0, :, :, :]
ortho = OS3D(img_no_pt, cmap="coolwarm").show()

sd = torch.load("models/brain-biomarker-site-v0_pious-galaxy-47/best_model.pth")["state_dict"]
model.load_state_dict(sd)

ig = IntegratedGradients(model)

attribution = ig.attribute(inp, target=1, n_steps=500)
img_pt = attribution.detach().numpy()[0, 0, :, :, :]
ortho = OS3D(img_pt, cmap="coolwarm").show()

compare_fm_differences(img_no_pt, img_pt)



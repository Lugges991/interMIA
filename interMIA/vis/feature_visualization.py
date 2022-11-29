import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import nibabel as nib
from interMIA.models import ResNet


torch.manual_seed(42)


def feature_visualization(pretrain=True):

    model = ResNet().cuda()
    if pretrain:
        sd = torch.load("models/brain-biomarker-site-v0_pious-galaxy-47/best_model.pth")["state_dict"]
        model.load_state_dict(sd)



    img = torch.rand((1, 2, 32, 32, 32)).cuda()
    optimizer = optim.Adam([img], lr=0.1)

    selected_layer = ""
    conv_output = 0
    selected_filter = 32


    for i in range(0, 10000):
        optimizer.zero_grad()

        x = img

        x = model.conv1(x)
        x = model.maxpool(x)
        x = model.layer0(x)
        # x = model.layer1(x)
        # x = model.layer2(x)
        conv_output = x[0, selected_filter]
        
        loss = -torch.mean(conv_output)
        loss.backward()
        optimizer.step()

        # if i % 20 == 0:
        #     plt.imshow(img[0, 0, :, :, 8].detach().cpu().numpy())
        #     plt.show()

    vol1 = nib.Nifti1Image(img[0, 0, :].detach().cpu().numpy(), affine=np.eye(4))
    nib.save(vol1, "data/feature_vis_0.nii")
    vol2 = nib.Nifti1Image(img[0, 1, :].detach().cpu().numpy(), affine=np.eye(4))
    nib.save(vol2, "data/feature_vis_1.nii")

    return vol1, vol2




if __name__ == "__main__":
    feature_visualization()

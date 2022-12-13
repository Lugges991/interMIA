import torch
from interMIA.models import ViT3D
from interMIA.dataloader import data_2c
from torch.utils.data import DataLoader
from captum.attr import LayerGradCam
from OrthoSlicer3D import OrthoSlicer3D as OS3D



val_data = data_2c("data/val.csv")
val_loader = DataLoader(
    val_data, batch_size=16, shuffle=True)


model = ViT3D(patch_size=16, heads=16, depth=24, dim=1024, mlp_dim=4096).cuda()

dl_iter = iter(val_loader)
x, y = next(dl_iter)

x = x.cuda()
y = y.cuda()

breakpoint()
# load model weights
sd = torch.load("models/brain-biomarker-whole-trafo-v0_copper-rain-1/model_epoch_8.pth")["state_dict"]
model.load_state_dict(sd)
out = model(x)


# visualize attention maps
lgc = LayerGradCam(model, model.patch_embedding)
attr = lgc.attribute(x, target=0)
img_pt = attr.detach().numpy()[0, 0, :, :, :]
ortho = OS3D(img_pt, cmap="coolwarm").show()


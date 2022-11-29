import torch
from interMIA.models import ResNet
from interMIA.vis import feature_visualization, ClassSpecificImageGeneration, compare_fm_differences

target_class = 1 # 0: Autism; 1: TC

model = ResNet()
csig = ClassSpecificImageGeneration(model=model, target_class=target_class)
img_no_pt = csig.generate()[0,0,:].detach().numpy()


sd = torch.load("models/brain-biomarker-site-v0_pious-galaxy-47/best_model.pth")["state_dict"]
model.load_state_dict(sd)

csig = ClassSpecificImageGeneration(model=model, target_class=target_class)
img_pre = csig.generate()[0,0,:].detach().numpy()

compare_fm_differences(img_no_pt, img_pre)

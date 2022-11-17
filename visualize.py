import torch
from interMIA.models import ResNet
from interMIA.vis import feature_visualization, ClassSpecificImageGeneration

target_class = 1 # 0: Autism; 1: TC

model = ResNet()
sd = torch.load("models/brain-biomarker-site-v0_pious-galaxy-47/best_model.pth")["state_dict"]
model.load_state_dict(sd)

csig = ClassSpecificImageGeneration(model=model, target_class=target_class)
csig.generate()

# feature_visualization()

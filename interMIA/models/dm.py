import torch
from torch import nn
from torchinfo import summary


class DM(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding="same"),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        self.c2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding="same"),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        self.c3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding="same"),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        self.c4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(2*2*2*128, 128))
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = self.c4(out)

        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out


if __name__ == "__main__":
    model = DM().to("cuda")
    inp = torch.rand((1, 2, 32, 32, 32)).to("cuda")
    # out = model(inp)
    # print(out.cpu().detach().numpy())
    summary(model, input_data=inp)

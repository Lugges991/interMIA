import torch
from torch import nn
from torchinfo import summary


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=True, k_size=3):
        super().__init__()

        layers = []
        for i in range(len(out_channels)):
            if i == 0:
                layers.append(nn.Conv3d(
                    in_channels=in_channels, out_channels=out_channels[i], kernel_size=k_size, padding="same"))
            else:
                layers.append(nn.Conv3d(
                    in_channels=out_channels[i-1], out_channels=out_channels[i], kernel_size=k_size, padding="same"))
            layers.append(nn.ReLU())

        if max_pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x


class TwoCVGG(nn.Module):
    def __init__(self, in_channels=2, n_classes=2):
        super().__init__()

        # input size = 32, 32, 32
        self.b1 = Block(in_channels=in_channels, out_channels=[64, 64])
        self.b2 = Block(in_channels=64, out_channels=[128, 128])
        self.b3 = Block(in_channels=128, out_channels=[256, 256, 256])
        self.b4 = Block(in_channels=256, out_channels=[512, 512, 512])
        self.b5 = Block(in_channels=512, out_channels=[512, 512, 512])

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(512, 4096), nn.Linear(4096, 4096), nn.Linear(4096, n_classes))


    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)

        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = TwoCVGG().to("cuda")
    inp = torch.rand((1, 2, 32, 32, 32)).to("cuda")
    # out = model(inp)
    # print(out.cpu().detach().numpy())
    summary(model, input_data=inp)

import torch
from torch import nn
from torchinfo import summary


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, p_kernel=(2, 2, 2)):
        super().__init__()

        c_kernels = [(3, 3, 3) for i in range(len(out_channels))]

        self.layer_list = []
        for i, (in_c, out_c) in enumerate(zip(in_channels, out_channels)):
            self.layer_list.append(
                nn.Conv3d(in_c, out_c, kernel_size=c_kernels[i]))
            self.layer_list.append(nn.ReLU())
            self.layer_list.append(nn.MaxPool3d(
                kernel_size=p_kernel, stride=1))

        self.layer_list = nn.ModuleList(self.layer_list)

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x


class TwoCC3D(nn.Module):
    def __init__(self, in_channels=2, n_classes=2):
        super().__init__()

        self.block1 = Block(in_channels=[in_channels], out_channels=[
                            32])
        # self.do1 = nn.Dropout(p=0.5)
        self.block2 = Block(in_channels=[32], out_channels=[64])
        # self.do2 = nn.Dropout(p=0.5)
        self.block3 = Block(in_channels=[64, 128], out_channels=[128, 128])
        # self.do3 = nn.Dropout(p=0.65)
        self.block4 = Block(in_channels=[128, 128], out_channels=[128, 128])
        # self.do4 = nn.Dropout(p=0.65)
        self.flat = nn.Flatten()

        self.fc6 = nn.Linear(128 * 14**3, 256)
        self.relu1 = nn.ReLU()
        self.fc7 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.fc8 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.block1(x)
        # x = self.do1(x)
        x = self.block2(x)
        # x = self.do2(x)
        x = self.block3(x)
        # x = self.do3(x)
        x = self.block4(x)
        # x = self.do4(x)
        x = self.flat(x)

        x = self.fc6(x)
        x = self.relu1(x)
        x = self.fc7(x)
        x = self.relu2(x)
        x = self.fc8(x)
        return x


if __name__ == "__main__":
    model = TwoCC3D().to("cuda")
    inp = torch.rand((1, 2, 32, 32, 32)).to("cuda")
    # out = model(inp)
    # print(out.cpu().detach().numpy())
    summary(model, input_data=inp)

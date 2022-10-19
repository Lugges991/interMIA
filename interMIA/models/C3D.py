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
            self.layer_list.append(nn.MaxPool3d(
                kernel_size=p_kernel, stride=1))

        self.layer_list = nn.ModuleList(self.layer_list)

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x


class C3D(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.block1 = Block(in_channels=[in_channels], out_channels=[
                            64], p_kernel=(1, 2, 2))
        self.block2 = Block(in_channels=[64], out_channels=[128])
        self.block3 = Block(in_channels=[128, 256], out_channels=[256, 256])
        self.block4 = Block(in_channels=[256, 512], out_channels=[512, 512])
        self.block5 = Block(in_channels=[512, 512], out_channels=[512, 512])
        self.flat = nn.Flatten()
        self.fc6 = nn.Linear(512 * 9 * 8 * 8, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, n_classes)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flat(x)

        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.sm(x)
        return x


if __name__ == "__main__":
    c3d = C3D(1, 2).to("cuda")
    inp = torch.rand((4, 1, 32, 32, 32)).to("cuda")
    print(c3d(inp).shape)
    summary(c3d, input_data=inp)

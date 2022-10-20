import torch
from torch import nn

class DarkNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nr_features, kernel=(2, 2, 2)):
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel)
        self.bn = nn.BatchNorm3d(nr_features)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)


class ThreeDNBlock(nn.Module):
    def __init__(self, ):
        pass

class DarkASDNet(nn.Module):
    def __init__(self, in_channels, n_classes, img_size=(32, 32, 32)):
        self.dn1 = DarkNetBlock()
        self.mp1 = nn.MaxPool3d()
        self.dn2 = DarkNetBlock()

        self.b1 = ThreeDNBlock()
        self.b2 = ThreeDNBlock()
        self.b3 = ThreeDNBlock()
        self.b4 = ThreeDNBlock()
        self.b5 = ThreeDNBlock()

        self.dn3 = DarkNetBlock()
        self.dn4 = DarkNetBlock()

        self.conv = nn.Conv3d()
        self.flat = nn.Flatten()
        self.lin = nn.Linear()

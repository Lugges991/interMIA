import torch
from torch import nn
from torchinfo import summary


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block=ResidualBlock, layers=[2, 2, 2], num_classes=2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=1)
        self.avgpool = nn.AvgPool3d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 16
        x = self.maxpool(x)  # 8
        x = self.layer0(x)  # 8
        x = self.layer1(x)  # 8
        x = self.layer2(x)  # 8

        x = self.avgpool(x)  # 2
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def __str__(self,):
        return "ResNet"


if __name__ == "__main__":
    model = ResNet(ResidualBlock, [2, 2, 2], 2).cuda()
    inp = torch.rand((1, 2, 32, 32, 32)).cuda()
    # out = model(inp)

    # print(out.shape)
    summary(model, input_data=inp)

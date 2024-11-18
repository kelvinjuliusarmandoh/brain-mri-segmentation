
# model_parts.py
import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv_block1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_block2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.relu(x)
        x = self.conv_block2(x)
        x = self.relu(x)
        return x

class DownSampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DownSampling, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        down = self.double_conv(x)
        pool = self.max_pool(down)
        return down, pool

class UpSampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpSampling, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels//2, 2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, encoder):
        x = self.up_conv(x)
        x = torch.cat([x, encoder], dim=1)
        x = self.double_conv(x)
        return x

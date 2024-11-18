
# unet.py
import torch.nn as nn
from modulars.model_parts import DoubleConv, DownSampling, UpSampling

class UNet(nn.Module):
    def __init__(self, input_shape: int=3, output_shape: int=1):
        super(UNet, self).__init__()
        self.enc1 = DownSampling(input_shape, 64)
        self.enc2 = DownSampling(64, 128)
        self.enc3 = DownSampling(128, 256)
        self.enc4 = DownSampling(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.dec4 = UpSampling(1024, 512)
        self.dec3 = UpSampling(512, 256)
        self.dec2 = UpSampling(256, 128)
        self.dec1 = UpSampling(128, 64)

        self.out = nn.Conv2d(64, output_shape, 1)

    def forward(self, x):
        down1, pool1 = self.enc1(x)
        down2, pool2 = self.enc2(pool1)
        down3, pool3 = self.enc3(pool2)
        down4, pool4 = self.enc4(pool3)

        bottleneck = self.bottleneck(pool4)

        dec4 = self.dec4(bottleneck, down4)
        dec3 = self.dec3(dec4, down3)
        dec2 = self.dec2(dec3, down2)
        dec1 = self.dec1(dec2, down1)

        out = self.out(dec1)
        return out


# -*- coding: utf-8 -*-
"""
Created on Tue May 27 19:58:41 2025

@author: ZML
"""
import torch
import torch.nn as nn


def CBR(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

'''
def CBR(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
'''
class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.enc1 = CBR(in_ch, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 不可复现
        self.up = nn.Upsample(scale_factor=2)
        self.dec4 = CBR(512+256, 256)
        self.dec3 = CBR(256+128, 128)
        self.dec2 = CBR(128+64, 64)
        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d4 = self.up(e4)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))
        d3 = self.up(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        return self.final(d2), None
    
'''
net = UNet()
x = torch.randn(2,3,224,224)
print(net(x).shape)
'''

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:17:52 2025

@author: fandos
"""

import math
import torch.nn as nn


class ResidualBlock(nn.Module):
    
    def __init__(self, features):
        
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(features)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x):
        
        input_tensor = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        return input_tensor + x
    
    
class UpsampleBlock(nn.Module):
    
    def __init__(self, channels, upscale_factor):
        
        super(UpsampleBlock, self).__init__()
        
        self.conv = nn.Conv2d(channels, channels * upscale_factor**2, kernel_size=3, stride=1, padding=1)
        self.pixelshuffler = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        
        x = self.conv(x)
        x = self.pixelshuffler(x)
        x = self.prelu(x)

        return x
    
    
class Generator(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=3, channels=64, n_resblocks=16, upscale_factor=2):
        
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()

        # Residual blocks
        res_blocks = []
        for _ in range(n_resblocks):
            res_blocks.append(ResidualBlock(channels))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

        # Upsampling layers
        upsampling = []
        for _ in range(int(math.log(upscale_factor, 2))):
            upsampling.append(UpsampleBlock(channels, 2))
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=9, stride=1, padding=4)
        self.tanh = nn.Tanh()
        
        # Initialize neural network weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.prelu(x)
        
        out_resblocks = self.res_blocks(x)
        
        out_resblocks = self.conv2(out_resblocks)
        out_resblocks = self.bn(out_resblocks)
        
        out = x + out_resblocks
        
        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.tanh(out)
        
        return out
    
    
class Discriminator(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1):
        
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.leaky_relu1 = nn.LeakyReLU(0.2, True)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.leaky_relu2 = nn.LeakyReLU(0.2, True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.leaky_relu3 = nn.LeakyReLU(0.2, True)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.leaky_relu4 = nn.LeakyReLU(0.2, True)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.leaky_relu5 = nn.LeakyReLU(0.2, True)
        
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.leaky_relu6 = nn.LeakyReLU(0.2, True)
        
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.leaky_relu7 = nn.LeakyReLU(0.2, True)
        
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.leaky_relu8 = nn.LeakyReLU(0.2, True)
        
        self.globalAvgPool = nn.AdaptiveAvgPool2d((15, 20))
        
        self.dense = nn.Linear(512 * 15 * 20, 1024)
        self.leaky_relu9 = nn.LeakyReLU(0.2, True)
        self.output = nn.Linear(1024, out_channels)

    def forward(self, x):
        
        x = self.leaky_relu1(self.conv1(x))
        
        x = self.leaky_relu2(self.bn2(self.conv2(x)))
        x = self.leaky_relu3(self.bn3(self.conv3(x)))
        x = self.leaky_relu4(self.bn4(self.conv4(x)))
        x = self.leaky_relu5(self.bn5(self.conv5(x)))
        x = self.leaky_relu6(self.bn6(self.conv6(x)))
        x = self.leaky_relu7(self.bn7(self.conv7(x)))
        x = self.leaky_relu8(self.bn8(self.conv8(x)))
        
        x = self.globalAvgPool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.dense(x)
        x = self.leaky_relu9(x)
        out = self.output(x)

        return out
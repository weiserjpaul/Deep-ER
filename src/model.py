import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mr_tools import torch_reco_img_to_kspace, torch_reco_kspace_to_img
from src.interlacer_layer import Interlacer, Mix


class ResidualInterlacer(nn.Module):
    def __init__(self, kernel_size,
                        num_features_img,
                        num_features_img_inter,
                        num_features_kspace,
                        num_convs,
                        num_layers,
                        use_cc,
                        use_norm):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_features_img = num_features_img
        self.num_features_img_inter = num_features_img_inter
        self.num_features_space = num_features_kspace
        self.num_convs = num_convs
        self.num_layers = num_layers
        self.use_cc = use_cc
        self.use_norm = use_norm
        
        self.interlacer_layers = nn.ModuleList([Interlacer(features_img=self.num_features_img,
                                  features_img_inter=self.num_features_img_inter,
                                  features_kspace=self.num_features_space,
                                  kernel_size=self.kernel_size,
                                  num_convs=self.num_convs,
                                  use_cc=self.use_cc,
                                  use_norm=self.use_norm,
                                 ) for i in range(self.num_layers)])
        
        
        
        self.conv1d_img = nn.Conv3d(in_channels=2*self.num_features_img,
                                   out_channels=self.num_features_img,
                                   kernel_size=1, #self.kernel_size,
                                   padding='same')
        self.conv1d_kspace = nn.Conv3d(in_channels=2*self.num_features_space,
                                       out_channels=self.num_features_space,
                                       kernel_size=1, #self.kernel_size,
                                       padding='same')
        

    def forward(self, x, hamming_grid, sense, perm, phase_coil, homCorr):
        inputs_img, inputs_kspace = x
        
        
        img_in = inputs_img
        freq_in = inputs_kspace
        
        for i in range(self.num_layers):
            img_conv, k_conv = self.interlacer_layers[i]((img_in, freq_in, inputs_img, inputs_kspace), hamming_grid, sense, perm, phase_coil, homCorr)
            
            img_in = img_conv + img_in #inputs_img
            freq_in = k_conv + freq_in #inputs_kspace

            
        
        
        
        outputs_img = self.conv1d_img(torch.cat((img_in, inputs_img), dim=1))
        outputs_kspace = self.conv1d_kspace(torch.cat((freq_in, inputs_kspace), dim=1))
        

        return (outputs_img, outputs_kspace)






class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mr_tools import torch_reco_img_to_kspace, torch_reco_kspace_to_img

def piecewise_relu(x):
    """Custom nonlinearity for freq-space convolutions."""
    return x + F.relu(1 / 2 * (x - 1)) + F.relu(1 / 2 * (-1 - x))


def get_nonlinear_layer(nonlinearity):
    """Selects and returns an appropriate nonlinearity."""
    if(nonlinearity == 'relu'):
        return torch.nn.ReLU()
    elif(nonlinearity == '3-piece'):
        return piecewise_relu

class BatchNormConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_norm):
        super().__init__()
        self.in_channels = in_channels
        
        self.in_channels_conv = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_norm = use_norm

        if use_norm=="BatchNorm":
            self.bn = nn.BatchNorm3d(
                num_features = self.in_channels)
        elif use_norm=="InstanceNorm":
            self.bn = nn.InstanceNorm3d(
                num_features = self.in_channels)
        elif use_norm=="None":
            self.bn = None
        else:
            print("Pick an available Normalizaiton Method")
            sys.exit()
        
        self.conv = nn.Conv3d(
            in_channels = self.in_channels_conv,
            out_channels = self.out_channels,
            kernel_size=self.kernel_size,
            padding="same")
    
    def forward(self, x):
        """Core layer function to combine BN/convolution.
        Args:
          x: Input tensor
        Returns:
          conv(float): Output of BN (on axis 0) followed by convolution
        """
        
        if not self.use_norm=="None":
            x = self.bn(x)
        x = self.conv(x)

        return x
    """
    def compute_output_shape(self, input_shape):
        return (input_shape[:3] + [self.features])
    """

class Mix(nn.Module):
    """Custom layer to learn a combination of two inputs."""
    def __init__(self):
        super().__init__()
        self._mix = nn.Parameter(torch.rand((1,)), requires_grad=True)
        
    def forward(self, x):
        """Core layer function to combine inputs.
        Args:
          x: Tuple (A,B), where A and B are numpy arrays of equal shape
        Returns:
          sig_mix*A + (1-sig_mix)B, where six_mix = sigmoid(mix) and mix is a learned combination parameter
        """
        A, B = x
        sig_mix = torch.sigmoid(self._mix)
        return sig_mix * A + (1 - sig_mix) * B
    
    """
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    """



class Interlacer(nn.Module):
    """Custom layer to learn features in both image and frequency space."""
    def __init__(self, features_img, features_img_inter, features_kspace, kernel_size, use_cc, 
                        use_norm, num_convs=1, shift=False):
        super().__init__()
        self.features_img = features_img
        self.features_img_inter = features_img_inter
        self.features_kspace = features_kspace
        self.kernel_size = kernel_size
        self.num_convs = num_convs
        self.use_cc = use_cc
        self.use_norm = use_norm
        
        self.img_mix = Mix()
        self.freq_mix = Mix()
        ImgModuleList = []
        FreqModuleList = []
        for i in range(self.num_convs):
            if i == 0:
                ImgModuleList.append(BatchNormConv(self.features_img*2, self.features_img, self.kernel_size, self.use_norm))
                FreqModuleList.append(BatchNormConv(self.features_kspace*2, self.features_kspace, self.kernel_size, self.use_norm))
            else:
                ImgModuleList.append(BatchNormConv(self.features_img, self.features_img, self.kernel_size, self.use_norm))
                FreqModuleList.append(BatchNormConv(self.features_kspace, self.features_kspace, self.kernel_size, self.use_norm))
        self.img_bnconvs = nn.ModuleList(ImgModuleList)
        self.freq_bnconvs = nn.ModuleList(FreqModuleList)
        
    def forward(self, x, hamming_grid, sense, perm, phase_coil, homCorr):
        """Core layer function to learn image and frequency features.
        Args:
          x: Tuple (A,B), where A contains image-space features and B contains frequency-space features
        Returns:
          img_conv(float): nonlinear(conv(BN(beta*img_in+IFFT(freq_in))))
          freq_conv(float): nonlinear(conv(BN(alpha*freq_in+FFT(img_in))))
        """
        
        img_in, freq_in, inputs_img, inputs_kspace = x

        
        batchsz = img_in.shape[0]
        for i in range(batchsz):
            if i == 0:
                img_in_as_freq = torch.unsqueeze(torch_reco_img_to_kspace(img_cc=img_in[i], 
                                                                            sense=sense[:,:,:,perm[i]]*phase_coil[i,None,None,None,:], 
                                                                            hamming_grid=hamming_grid, 
                                                                            homCorr=homCorr,
                                                                            use_cc=self.use_cc), 
                                                dim=0)
                freq_in_as_img = torch.unsqueeze(torch_reco_kspace_to_img(k_coils=freq_in[i],
                                                                            sense=sense[:,:,:,perm[i]]*phase_coil[i,None,None,None,:], 
                                                                            hamming_grid=hamming_grid, 
                                                                            homCorr=homCorr,
                                                                            use_cc=self.use_cc), 
                                                dim=0)
            else:
                img_in_as_freq = torch.cat((img_in_as_freq, torch.unsqueeze(torch_reco_img_to_kspace(img_cc=img_in[i],
                                                                                                    sense=sense[:,:,:,perm[i]]*phase_coil[i,None,None,None,:], 
                                                                                                    hamming_grid=hamming_grid, 
                                                                                                    homCorr=homCorr,
                                                                                                    use_cc=self.use_cc), 
                                                                            dim=0)
                                            ), 
                                            dim=0)
                freq_in_as_img = torch.cat((freq_in_as_img, torch.unsqueeze(torch_reco_kspace_to_img(k_coils=freq_in[i],
                                                                                                    sense=sense[:,:,:,perm[i]]*phase_coil[i,None,None,None,:], 
                                                                                                    hamming_grid=hamming_grid, 
                                                                                                    homCorr=homCorr,
                                                                                                    use_cc=self.use_cc), 
                                                                            dim=0)
                                            ), 
                                            dim=0)

        img_feat = self.img_mix([img_in, freq_in_as_img])
        k_feat = self.freq_mix([freq_in, img_in_as_freq])

        img_feat = torch.cat((img_feat, inputs_img), dim=1)
        k_feat = torch.cat((k_feat, inputs_kspace), dim=1)    
        
        for i in range(self.num_convs):
            img_conv = self.img_bnconvs[i](img_feat)
            img_feat = get_nonlinear_layer('relu')(img_conv)

            k_conv = self.freq_bnconvs[i](k_feat)
            k_feat = get_nonlinear_layer('3-piece')(k_conv)
        

        return (img_feat, k_feat)



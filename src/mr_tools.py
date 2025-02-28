import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_reco_img_to_kspace(img_cc, sense, hamming_grid=None, homCorr=False, coils=32, use_cc=True):
    
    if use_cc:
        if img_cc.dtype not in [torch.chalf, torch.cfloat, torch.cdouble]:
            img_cc = img_cc[0]+1j*img_cc[1]
        if torch.any(homCorr):
            img_cc = img_cc*homCorr
        img_coils = (torch.unsqueeze(img_cc, dim=3) * sense)
    else:
        if img_cc.dtype not in [torch.chalf, torch.cfloat, torch.cdouble]:
            img_cc = img_cc[:coils]+1j*img_cc[coils:]
        img_coils = torch.moveaxis(img_cc, 0,3)
    k_coils = torch.fft.fftshift(torch.fft.fft(img_coils, dim=2),dim=2)
    k_coils = torch.fft.fftshift(torch.fft.fft(k_coils, dim=1),dim=1)
    k_coils = torch.fft.fftshift(torch.fft.fft(k_coils, dim=0),dim=0)
    if torch.any(hamming_grid):
        k_coils = k_coils/hamming_grid[:,:,:,None]
    k_coils = torch.moveaxis(k_coils, 3,0)
    k_coils = torch.cat((torch.real(k_coils), torch.imag(k_coils)),dim=0)
    
    return k_coils


def torch_reco_kspace_to_img(k_coils, sense, hamming_grid=None, homCorr=False, coils=32, brainMask=False, use_cc=True, final=False): 
    k_coils = k_coils[:coils]+1j*k_coils[coils:]
    img_grid_fft = torch.moveaxis(k_coils, 0,-1)
    
    if torch.any(hamming_grid):
        img_grid_fft = img_grid_fft*hamming_grid[:,:,:,None]
    img_grid_fft = torch.fft.ifft(torch.fft.ifftshift(img_grid_fft, dim=0), dim=0)
    img_grid_fft = torch.fft.ifft(torch.fft.ifftshift(img_grid_fft, dim=1), dim=1)
    img_grid_fft = torch.fft.ifft(torch.fft.ifftshift(img_grid_fft, dim=2), dim=2)
    if use_cc:
        img_grid_cc = torch.sum(torch.conj(sense)*img_grid_fft, dim=3)
        if False:
            img_grid_cc = img_grid_cc/homCorr
        img_grid_cc = torch.stack((torch.real(img_grid_cc), torch.imag(img_grid_cc)),dim=0)
    else:
        if final:
            img_grid_cc = torch.sum(torch.conj(sense)*img_grid_fft, dim=3)
            #img_grid_cc = img_grid_cc/homCorr
            #img_grid_cc = img_grid_cc*brainMask
            img_grid_cc = torch.stack((torch.real(img_grid_cc), torch.imag(img_grid_cc)),dim=0)
        else:
            img_grid_cc = torch.moveaxis(img_grid_fft, 3,0)
            img_grid_cc = torch.cat((torch.real(img_grid_cc), torch.imag(img_grid_cc)),dim=0)
    
    return img_grid_cc

def reco_nonCart_to_Img_torch(mrsiData_kz, hamming_kz, dft_inv_kz, voronoi, sense, homCorr, device, coils=32, use_cc=True, kz_min=-15, kz_max=15, mrsi=False):
    img=torch.zeros((64,64,31,coils), dtype=torch.cfloat).to(device)
    for k_z in range(kz_min,kz_max+1):
        ind_z = (k_z%31)
        tmp = hamming_kz[str(ind_z)][:,None] * mrsiData_kz[str(ind_z)]
        tmp = torch.matmul(torch.diag(voronoi[str(ind_z)]), tmp)
        tmp = torch.matmul(dft_inv_kz[str(ind_z)], tmp)
        img[:,:,ind_z,:] = torch.reshape(tmp, (64,64,coils))
        
    img_ifft = torch.fft.ifft(img, dim=2)
    if mrsi:
        img_ifft = torch.fft.ifftshift(img_ifft, dim=2)
    if use_cc:
        img_cc = torch.sum(torch.conj(sense)*img_ifft, dim=3)/homCorr
        if False:
            print("img1: ", img_ifft[32,32,15,15])
            img_coils = img_cc*homCorr
            img_coils = (torch.unsqueeze(img_coils, dim=3) * sense)
            print("img2: ", img_coils[32,32,15,15])
            xxx = torch.sum(torch.conj(sense)*sense, dim=3)
            print("diff1: ", (img_ifft-img_coils)[:,32,15,15])
            print("homCorr: ", homCorr[:,32,15])
            print("sense: ", xxx[:,32,15])
    else:
        #mask = torch.sum(sense*torch.conj(sense), dim=-1)
        #img_ifft = torch.unsqueeze(mask, dim=-1)*img_ifft
        if False:
            img_ifft = img_ifft/torch.unsqueeze(homCorr, dim=3)
        img_cc = torch.moveaxis(img_ifft, 3,0)
    return img_cc

def reco_nonCart_to_Img_torch_noVoronoi(mrsiData_kz, hamming_kz, dft_inv_kz, sense, homCorr, device, coils=32, use_cc=True, kz_min=-15, kz_max=15):
    img=torch.zeros((64,64,31,coils), dtype=torch.cfloat).to(device)
    for k_z in range(kz_min,kz_max+1):
        ind_z = (k_z%31)
        tmp = hamming_kz[str(ind_z)][:,None] * mrsiData_kz[str(ind_z)]
        tmp = torch.matmul(dft_inv_kz[str(ind_z)], tmp)
        img[:,:,ind_z,:] = torch.reshape(tmp, (64,64,coils))
        
    img_ifft = torch.fft.ifft(img, dim=2)
    if use_cc:
        img_cc = torch.sum(torch.conj(sense)*img_ifft, dim=3)/homCorr
    else:
        img_cc = torch.moveaxis(img_ifft, 3,0)
    return img_cc



def SamplingDensityVoronoi(K,MaxRad=None):
    from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
    # Compute Density Compensation Based on Voronoi polygon
    # K = K - np.mean(K,axis=0)
    if (MaxRad is None):
        MaxRad=np.max(np.sqrt(np.power(K[:,0],2)+ np.power(K[:,1],2)))
    #uK = np.unique(K, axis=0)
    vor = Voronoi(K)
    
    #fig = voronoi_plot_2d(vor)
    #plt.show()
    lg_indices = np.where(np.linalg.norm(vor.vertices, axis=1) > MaxRad)[0]

    VArea = np.zeros(vor.npoints)
    for i, reg_num in enumerate(vor.point_region):
        indices = vor.regions[reg_num]
        if -1 in indices or len(set(indices).intersection(lg_indices)) >0: # some regions can be opened
            VArea[i] = np.inf
        else:
            VArea[i] = ConvexHull(vor.vertices[indices]).volume


    if np.sum(np.logical_not(np.isinf(VArea.reshape(-1)))) == 0:
        maxVArea = 1 #all VArea are NaN, there is no max, set values to 1
    else:
        maxVArea = np.max(VArea[np.logical_not(np.isinf(VArea))])
    
    VArea[np.isinf(VArea)] = maxVArea # basically assign value equivalent to the k-space center to all outer points for which domain computation was not possible
    
    return VArea

def create_brute_force_FT_matrix_torch(In, GridX, GridY, device):

    N1 = GridX.shape[0]

    NN1 = In[0]
    NN2 = In[1]

    #XX, YY = torch.meshgrid(torch.arange(-NN1/2+1,NN1/2+1)/NN1, torch.arange(-NN2/2+1,NN2/2+1)/NN2)
    #YY, XX = torch.meshgrid(torch.arange(-NN1/2+1,NN1/2+1)/NN1, torch.arange(-NN2/2+1,NN2/2+1)/NN2) # Numpy and Pytorch have switched output with meshgrid
    YY, XX = torch.meshgrid(torch.arange((-NN1+1)/2,(NN1+1)/2)/NN1, torch.arange((-NN2+1)/2,(NN2+1)/2)/NN2)
    
    XX = torch.reshape(XX, (1,-1)).to(device)
    YY = torch.reshape(YY, (1,-1)).to(device)
    
    GridX=GridX[:,None]
    GridY=GridY[:,None]
    
    
    W = torch.exp(-2*np.pi*1j*(GridX*XX + GridY*YY))
    W = W / np.sqrt(XX.shape[1])
    W = W.type(torch.complex64)
    return W
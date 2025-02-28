import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchmetrics import StructuralSimilarityIndexMeasure
import sys

def get_opimizer(params, model):
    if params["optimizer_tag"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=params["lr"])
    else:
        return None


def get_loss(loss_tag, loss_space, device):
    
    def loss_func(reco_img, img, img_under, reco_kspace, mrsiData_Cart, mrsiData_Cart_under, mask, mrsi):

        img_space_max = torch.amax(img, dim=(1,2,3,4))[:,None,None,None,None]
        k_space_max = torch.amax(mrsiData_Cart, dim=(1,2,3,4))[:,None,None,None,None]

        if loss_space == "img" or loss_space == "img_rel" or loss_space == "both":
            reco_img, img, img_under = reco_img/img_space_max, img/img_space_max, img_under/img_space_max
        if loss_space == "k" or loss_space == "both":
            reco_kspace, mrsiData_Cart, mrsiData_Cart_under = reco_kspace/k_space_max, mrsiData_Cart/k_space_max, mrsiData_Cart_under/k_space_max

        if loss_tag == "mse":
            if loss_space == "img_vrel":
                loss_function = nn.MSELoss(reduction='none')
            else:
                loss_function = nn.MSELoss()
        elif loss_tag == "ncc":
            ncc = NCC().loss
            loss_function = lambda x,y : ncc(x,y) + 1
        elif loss_tag == "ssim":
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            loss_function = lambda x,y : -ssim(x,y)+1
        elif loss_tag == "ssimse":
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            mse = nn.MSELoss()
            loss_function = lambda x,y : -ssim(x,y)+1 + mse(x,y)
        elif loss_tag == "4ssimse":
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            mse = nn.MSELoss()
            loss_function = lambda x,y : 4*(-ssim(x,y)+1) + mse(x,y)
        elif loss_tag == "001ssimse":
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            mse = nn.MSELoss()
            loss_function = lambda x,y :  0.1*(-ssim(x,y)+1) +10*mse(x,y) #
        else:
            print("Incomplete loss tag definition")

        if mrsi:
            reco_img = reco_img*mask[:,None,:,:,:].to(device)
            img = img*mask[:,None,:,:,:].to(device)

        if loss_space == "img":
            loss = loss_function(reco_img, img)
        elif loss_space == "log_img":
            reco_imgc=torch.tile(reco_img[:,0,None]+1j*reco_img[:,1,None], (1,2,1,1,1))
            reco_mask=reco_imgc!=0

            reco_log = torch.clone(reco_img)
            reco_log[reco_mask]=(reco_img[reco_mask]*torch.log(torch.abs(reco_imgc[reco_mask])+1))/torch.abs(reco_imgc[reco_mask])
            
            imgc=torch.tile(img[:,0,None]+1j*img[:,1,None], (1,2,1,1,1))
            img_mask=imgc!=0
            img_log = torch.clone(img)
            img_log[img_mask]=(img[img_mask]*torch.log(torch.abs(imgc[img_mask])+1))/torch.abs(imgc[img_mask])
    
            loss = loss_function(reco_log, img_log)
        elif loss_space == "img_rel":
            loss = loss_function(reco_img, img) / torch.maximum(loss_function(img_under, img), torch.tensor(1e-5))
        elif loss_space == "img_vrel":
            loss = torch.mean(loss_function(reco_img, img) / torch.maximum(loss_function(img_under, img), torch.tensor(1e-5)))
        elif loss_space == "k":
            loss = loss_function(reco_kspace, mrsiData_Cart)
        elif loss_space == "both":
            loss_img = loss_function(reco_img, img)
            loss_kspace = loss_function(reco_kspace, mrsiData_Cart)
            loss = loss_img + loss_kspace
        else:
            print("Incomplete loss space definition")
        return loss
    return loss_func

def compute_kspace_loss(reco_kspace, mrsiData_Cart, mrsiData_Cart_under):
    non_mask = mrsiData_Cart!=mrsiData_Cart_under
    if torch.any(non_mask):
        loss = loss_function(reco_kspace[non_mask], mrsiData_Cart[non_mask])
    else:
        loss = 0
    return loss

def h5group_to_dict(group):
    new_dict = {}
    for key in list(group.keys()):
        new_dict[key]=group[key]
    return new_dict

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, eps=1e-5 ,win=None, preserve_batch=False):
        self.win = win
        self.eps = torch.tensor(eps)
        self.preserve_batch = preserve_batch

    def loss(self, y_true, y_pred):
        #print("y_true", y_true.shape)
        #print("y_pred", y_pred.shape)

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, nb_feats, *vol_shape]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win
        #print("win: ", win)
        # compute filters
        in_ch = Ji.shape[1]
        sum_filt = torch.ones([1, in_ch, *win]).to(Ji.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)
        

        win_size = np.prod(win) * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = torch.maximum(cross, self.eps)
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = torch.maximum(I_var, self.eps)
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = torch.maximum(J_var, self.eps)

        cc = cross * cross / (I_var * J_var)

        if self.preserve_batch:
            cc = torch.flatten(cc, start_dim=1, end_dim=-1)
            return -torch.mean(cc, dim=-1)
        else:
            return -torch.mean(cc)


def get_timepoints(hf):
    if "mrsiData_Cart" in hf.keys():
        timepts = hf["mrsiData_Cart"].shape[-1]
    elif "img_lr_rrrt" in hf.keys():
        timepts = hf["img_lr_rrrt"].shape[-1]
    elif "img_lr_rrrt_2_0" in hf.keys():
        timepts = hf["img_lr_rrrt_2_0"].shape[-1]
    elif "img_lr_rrrt_1_0" in hf.keys():
        timepts = hf["img_lr_rrrt_1_0"].shape[-1]
    elif "mrsiData_nonCart" in hf.keys():
        timepts = hf["mrsiData_nonCart"].shape[-1]
    elif "mrsiData_nonCart_tkc" in hf.keys():
        timepts = hf["mrsiData_nonCart_tkc"].shape[0]
    else:
        print("Key not found")
        sys.exit()

    return timepts
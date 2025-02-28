import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import random
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src. dataloaderBatchNonCart import WaterBatchDatasetNonCart, WaterBatchSamplerNonCart, preprocessNonCart, postprocessNonCart, unfoldBatch, load_data
from src.training_tools import get_timepoints

def testing(params, model, files, do, device, epoch=None, writer=None, name=None):

    if do == "test":
        lbl = "test"
        printname = "++++ Testing +++"
        fname = "_T_"
    if do == "val":
        lbl = "validation"
        printname = "++++ Validation +++"
        fname = "_V_"
    if do == "val_epoch":
        lbl = "validation_" + str(epoch)
        printname = "++++ Validation Epoch "+str(epoch)+" +++"
        fname = "_V"+str(epoch)+"_"
    
    acceleration = [2,3,4,5]

    print(printname)
    for p_iter, p in enumerate(files):
        for acc in acceleration:
            print("++ " + p + ", Acceleration: " + str(acc) + " ++")

            validation_batchSampler = WaterBatchSamplerNonCart(path_to_data=params["path_to_data"],
                                                        files=[p], 
                                                        batch_size=params["batch_size"],
                                                        undersampleTime=False,
                                                        shuffle=False)

            validation_data = WaterBatchDatasetNonCart(imgsz=params['imgsz'],
                                        params=params,
                                        acc=acc,
                                        seed=True)

            validation_dataloader = DataLoader(validation_data,
                                        num_workers=params["workers"],
                                        sampler=validation_batchSampler)
            
            hf = h5py.File(params["path_to_data"] + p, 'r')
            timepts = get_timepoints(hf)
            reconstruction = torch.zeros((2,) + params["imgsz"] + (timepts,), dtype=torch.float32)
            ground_truth = torch.zeros((2,) + params["imgsz"] + (timepts,), dtype=torch.float32)
            input = torch.zeros((2,) + params["imgsz"] + (timepts,), dtype=torch.float32)

            model.eval()
            with torch.no_grad():

                val_loss = []
                sta_epoch = time.time()

                if params["n_batches_val"] == -1 or params["n_batches_val"] > len(validation_dataloader):
                    n_batches_val_tmp = len(validation_dataloader)
                else:
                    n_batches_val_tmp = params["n_batches_val"]

                for i, batch in enumerate(tqdm(validation_dataloader)):
                    if i >= n_batches_val_tmp:
                        break
                    sta_batch = time.time()

                    mrsiData_nonCart, img_tgv, Ind_kz, voronoi, voronoi_acc, GridX, GridY, hamming_kz, homCorr, \
                        sense, sense_ext, hamming_grid, homCorr_grid, tindex, H_u, mrsi, brainMask, sto_time  = unfoldBatch(batch)
                    img_under, mrsiData_Cart_under, img_pre, mrsiData_Cart, sense, sense_ext, perm, hamming_grid, homCorr, homCorr_grid, \
                        img_max, phase_coil, brainMask, brainMaskAug = preprocessNonCart(mrsiData_nonCart=mrsiData_nonCart, 
                                                                                    img_tgv=img_tgv,
                                                                                    Ind_kz=Ind_kz, 
                                                                                    voronoi=voronoi,
                                                                                    voronoi_acc=voronoi_acc,
                                                                                    GridX=GridX, 
                                                                                    GridY=GridY,
                                                                                    hamming_kz=hamming_kz, 
                                                                                    homCorr=homCorr, 
                                                                                    sense=sense, 
                                                                                    sense_ext=sense, #sense, 
                                                                                    hamming_grid=hamming_grid, 
                                                                                    homCorr_grid=homCorr_grid, 
                                                                                    tindex=tindex, 
                                                                                    params=params, 
                                                                                    b_coilPhase=False, 
                                                                                    b_globPhase=False, 
                                                                                    b_noise=False, 
                                                                                    b_knoise=False,
                                                                                    b_aug=False,
                                                                                    b_coilPerm=False,
                                                                                    b_tgv=params["use_tgv"],
                                                                                    mrsi=mrsi,
                                                                                    brainMask=brainMask,
                                                                                    device=device)
                    if False:
                        print("input-time: ", tindex)
                        print("input-img: ", torch.amax(torch.abs(img_under[:,0] + 1j*img_under[:,1]), dim=(1,2,3)))
                        print("input-kspace: ", torch.amax(torch.abs(mrsiData_Cart_under[:,0] + 1j*mrsiData_Cart_under[:,1]), dim=(1,2,3)))
                        
                    reco_img, reco_kspace = model((img_under, mrsiData_Cart_under), 
                                                    hamming_grid, 
                                                    sense, #sense,
                                                    perm,
                                                    phase_coil,
                                                    homCorr_grid,
                                                    H_u.to(device))
                    if False:
                        print("output1-reco: ", torch.amax(torch.abs(reco_img[:,0] + 1j*reco_img[:,1]), dim=(1,2,3)))
                        print("output1-kspce: ", torch.amax(torch.abs(reco_kspace[:,0] + 1j*reco_kspace[:,1]), dim=(1,2,3)))

                    reco_img, img, img_under = postprocessNonCart(reco_kspace=reco_kspace, 
                                                                mrsiData_Cart=mrsiData_Cart,
                                                                mrsiData_Cart_under=mrsiData_Cart_under,
                                                                sense=sense, 
                                                                perm=perm,
                                                                phase_coil=phase_coil, 
                                                                hamming_grid=hamming_grid, 
                                                                homCorr=homCorr, 
                                                                brainMask=brainMask,
                                                                brainMaskAug=brainMaskAug,
                                                                params=params)
                    if False:
                        print("output2, under: ", torch.amax(torch.abs(img_under[:,0] + 1j*img_under[:,1]), dim=(1,2,3)))
                        print("output2, reco: ", torch.amax(torch.abs(reco_img[:,0] + 1j*reco_img[:,1]), dim=(1,2,3)))
                        print("output2, true: ", torch.amax(torch.abs(img[:,0] + 1j*img[:,1]), dim=(1,2,3)))
                    
                    loss = params["loss_func"](reco_img=reco_img, 
                                                img=img,
                                                img_under=img_under,
                                                reco_kspace=reco_kspace, 
                                                mrsiData_Cart=mrsiData_Cart, 
                                                mrsiData_Cart_under=mrsiData_Cart_under,
                                                mrsi=mrsi,
                                                mask=brainMaskAug)

                    val_loss.append(loss.item())

                    sto_batch = time.time() - sta_batch
                    print("reco_img: ", reco_img.shape)
                    reco_img = reco_img*img_max[:,None,None,None,None]
                    img = img*img_max[:,None,None,None,None]
                    img_under = img_under*img_max[:,None,None,None,None]
                    reconstruction[:,:,:,:,tindex.long()] =  torch.moveaxis(reco_img.detach().cpu(), 0, -1)
                    ground_truth[:,:,:,:,tindex.long()] =  torch.moveaxis(img.detach().cpu(), 0, -1)
                    input[:,:,:,:,tindex.long()] =  torch.moveaxis(img_under.detach().cpu(), 0, -1)
                    

                val_loss = np.mean(np.array(val_loss))

                sto_epoch = time.time() - sta_epoch
                
                log_epoch = 'Loss: {:.8f}, GPU-Time: {:.4f}, Load-Time: {:.4f}'
                print(log_epoch.format(val_loss , sto_epoch, sto_time))
                print("\n")
                
                reconstruction = torch.flip(torch.rot90(reconstruction, k=1, dims=[1,2]), dims=(1,))
                ground_truth = torch.flip(torch.rot90(ground_truth, k=1, dims=[1,2]), dims=(1,))
                input = torch.flip(torch.rot90(input, k=1, dims=[1,2]), dims=(1,))
                
                if True:
                    hf = h5py.File(params["path_to_predictions"]+ "RECO2"  + "_" + name + fname + "ACC_" + str(acc) + "_" + p, 'w')
                    hf.create_dataset('reco', data=reconstruction)
                    hf.create_dataset('true', data=ground_truth)
                    hf.create_dataset('input', data=input)
                    hf.close()
                
                if writer:
                    writer.add_scalar(params["model_name"] + "/" + lbl + "_acc_" + str(acc), val_loss, p_iter)
                    writer.flush()
            
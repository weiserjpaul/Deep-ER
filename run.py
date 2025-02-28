import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import random
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from config import params
from src.dataloaderBatchNonCart import unfoldBatch, WaterBatchSamplerNonCart, WaterBatchDatasetNonCart, load_data
from src.model import ResidualInterlacer
from src.mr_tools import torch_reco_img_to_kspace, torch_reco_kspace_to_img
from src.initialize import intialize_model_folder, save_params, read_args
from src.training_tools import get_opimizer, get_loss
from src.training import training, validation
from src.testing import testing





#### Initialize ####

params = read_args(params)

device = torch.device(params["gpu"] if torch.cuda.is_available() else 'cpu')
print("Using decive: ", device)

intialize_model_folder(params)

writer =  SummaryWriter(params["path_to_writer"] + "logs")

save_params(params)


#### Dataloader ####

training_batchSampler = WaterBatchSamplerNonCart(path_to_data=params["path_to_data"],
                                          files=params['train_paths'], 
                                          batch_size=params["batch_size"],
                                          undersampleTime=params["b_undersampleTime"],
                                          shuffle=True,
                                          ntimepts=params["ntimepts"],
                                          ntimepts3T=params["ntimepts3T"],
                                          oversampleTime=params["b_oversampleTime"])

training_data = WaterBatchDatasetNonCart(imgsz=params['imgsz'],
                                        params=params,
                                        acc="random", 
                                        seed=False)

train_dataloader = DataLoader(training_data,
                              num_workers=params["workers"],
                              sampler=training_batchSampler)


if len(params['val_water7T_paths']) > 0:
    val_water7T_dataloaders = []
    for path in params['val_water7T_paths']:
        validation_water7T_batchSampler = WaterBatchSamplerNonCart(path_to_data=params["path_to_data"], 
                                                    files=[path], 
                                                    batch_size=params["batch_size"],
                                                    undersampleTime=params["b_undersampleTime"],
                                                    shuffle=False,
                                                    ntimepts=params["ntimepts"],
                                                    ntimepts3T=params["ntimepts3T"])

        validation_water7T_data = WaterBatchDatasetNonCart(imgsz=params['imgsz'],
                                                    params=params,
                                                    acc="random", 
                                                    seed=False)

        val_water7T_dataloaders.append(DataLoader(validation_water7T_data,
                                                num_workers=params["workers"],
                                                sampler=validation_water7T_batchSampler))

if len(params['val_water3T_paths']) > 0:
    validation_water3T_batchSampler = WaterBatchSamplerNonCart(path_to_data=params["path_to_data"], 
                                                files=params['val_water3T_paths'], 
                                                batch_size=params["batch_size"],
                                                undersampleTime=params["b_undersampleTime"],
                                                shuffle=False,
                                                ntimepts=params["ntimepts"],
                                                ntimepts3T=params["ntimepts3T"])

    validation_water3T_data = WaterBatchDatasetNonCart(imgsz=params['imgsz'],
                                                params=params,
                                                acc="random", 
                                                seed=False)

    val_water3T_dataloader = DataLoader(validation_water3T_data,
                                num_workers=params["workers"],
                                sampler=validation_water3T_batchSampler)

if len(params['val_mrsi7T_paths']) > 0:
    val_mrsi7T_dataloaders = []
    for path in params['val_mrsi7T_paths']:
        for thisTime in [None]+params["thisTimepts"]:
            validation_mrsi7T_batchSampler = WaterBatchSamplerNonCart(path_to_data=params["path_to_data"], 
                                                        files=[path], 
                                                        batch_size=params["batch_size"],
                                                        undersampleTime=params["b_undersampleTime"],
                                                        shuffle=False,
                                                        ntimepts=params["ntimepts"],
                                                        ntimepts3T=params["ntimepts3T"],
                                                        thisTimepts=thisTime)

            validation_mrsi7T_data = WaterBatchDatasetNonCart(imgsz=params['imgsz'],
                                                        params=params,
                                                        acc="random", 
                                                        seed=False)

            val_mrsi7T_dataloaders.append(DataLoader(validation_mrsi7T_data,
                                                    num_workers=params["workers"],
                                                    sampler=validation_mrsi7T_batchSampler))

if len(params['val_mrsi3T_paths']) > 0:
    validation_mrsi3T_batchSampler = WaterBatchSamplerNonCart(path_to_data=params["path_to_data"], 
                                                files=params['val_mrsi3T_paths'], 
                                                batch_size=params["batch_size"],
                                                undersampleTime=params["b_undersampleTime"],
                                                shuffle=False,
                                                ntimepts=params["ntimepts"],
                                                ntimepts3T=params["ntimepts3T"])

    validation_mrsi3T_data = WaterBatchDatasetNonCart(imgsz=params['imgsz'],
                                                params=params,
                                                acc="random", 
                                                seed=False)

    val_mrsi3T_dataloader = DataLoader(validation_mrsi3T_data,
                                num_workers=params["workers"],
                                sampler=validation_mrsi3T_batchSampler)

#### Model ####

model = ResidualInterlacer(kernel_size=3,
                        num_features_img=params["num_features_img"],
                        num_features_img_inter=params["num_features_img_inter"],
                        num_features_kspace=64,
                        num_convs=4,
                        num_layers=20,
                        use_cc=params["use_cc"],
                        use_norm=params["use_norm"])
model.to(device)

optimizer = get_opimizer(params, model)

params["loss_func"] = get_loss(loss_tag = params["loss_tag"],
                                loss_space = params["loss_space"],
                                device = device)



#### Training ####
if params["b_train"]:
    best_loss = 0
    if params["load_model"]:
        print("loading model: " + params["load_exp_name"] + "/" + params["load_model_name"])
        load_path = params["path"] + "reco/models/" + params["load_exp_name"] + "/" + params["load_model_name"]
        model.load_state_dict(torch.load(load_path))

    for epoch in range(params["epochs"]):
        
        model = training(model = model,
                        train_dataloader = train_dataloader,
                        optimizer = optimizer,
                        params = params,
                        epoch = epoch,
                        device = device,
                        writer = writer)
        
        if len(params['val_water7T_paths']) > 0:
            val_water7T_epoch_loss = 0
            for d,dataloader in enumerate(val_water7T_dataloaders):
                val_water7T_epoch_loss += validation(model = model,
                                        validation_dataloader = dataloader,
                                        params = params,
                                        epoch = epoch,
                                        device = device,
                                        writer = writer,
                                        tag = "water7T_D"+str(d))
                    
                val_water7T_epoch_loss /= len(val_water7T_dataloaders)
        
        if len(params['val_water3T_paths']) > 0:
            val_water3T_epoch_loss = validation(model = model,
                                validation_dataloader = val_water3T_dataloader,
                                params = params,
                                epoch = epoch,
                                device = device,
                                writer = writer,
                                tag = "water3T")
        
        if len(params['val_mrsi7T_paths']) > 0:
            val_mrsi7T_epoch_loss = 0
            for d,dataloader in enumerate(val_mrsi7T_dataloaders):
                thisTimeInd=int(d%(len(params["thisTimepts"])+1))
                d=int(d/(len(params["thisTimepts"])+1))
                if thisTimeInd==0:
                    tag="metab7T_D"+str(d)+"_Tall"
                else:
                    tag="metab7T_D"+str(d)+"_T"+str(params["thisTimepts"][thisTimeInd-1])
                val_mrsi7T_epoch_loss += validation(model = model,
                                                    validation_dataloader = dataloader,
                                                    params = params,
                                                    epoch = epoch,
                                                    device = device,
                                                    writer = writer,
                                                    tag =tag)
                if thisTime==None:
                    val_mrsi7T_epoch_loss /= len(val_mrsi7T_dataloaders)
            
        
        if len(params['val_mrsi3T_paths']) > 0:
            val_mrsi3T_epoch_loss = validation(model = model,
                                    validation_dataloader = val_mrsi3T_dataloader,
                                    params = params,
                                    epoch = epoch,
                                    device = device,
                                    writer = writer,
                                    tag = "mrsi3T")

        this_epoch_loss = val_mrsi7T_epoch_loss
        #this_epoch_loss = val_water7T_epoch_loss
        if this_epoch_loss < best_loss or best_loss == 0:
            if params["save_model"]:
                best_loss = this_epoch_loss
                torch.save(model.state_dict(), params["path_to_model"] + "model_best.pt")
                f = open(params["path_to_model"] + "params.txt", "a")
                f.write("Save a Best Model in epoch " + str(epoch) + " with validation loss of " + str(this_epoch_loss))
                f.write('\n')
                f.close()
        if epoch%10 == 0:
            torch.save(model.state_dict(), params["path_to_model"] + "model_last.pt")
        



#### Testing ####
if params["b_test"]:
    for val_paths in [params['val_water7T_paths'], params['test_mrsi7T_paths'], params['val_water3T_paths'], params['val_mrsi3T_paths']]:
        
        if os.path.exists(params["path_to_model"] + "model_last.pt"):
            print("loading model: " + params["path_to_model"] + "model_last.pt")
            model.load_state_dict(torch.load(params["path_to_model"] + "model_last.pt"))

            testing(params = params, 
                model = model,
                files = val_paths, 
                do = "val",
                device = device,
                writer = writer,
                name = "LAST")


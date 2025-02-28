import os
import sys
import shutil
import numpy as np
import torch
import math, copy, time
import csv
import random
import argparse
from datetime import datetime


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_args(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--cc")
    parser.add_argument("--undersample_time")
    parser.add_argument("--aug")
    parser.add_argument("--aug_phase")
    parser.add_argument("--aug_noise")
    parser.add_argument("--aug_coilPerm")
    parser.add_argument("--b0_corr")
    parser.add_argument("--lowrank")
    parser.add_argument("--acc_low")
    parser.add_argument("--acc_high")
    parser.add_argument("--loss_space")
    parser.add_argument("--loss_tag")
    parser.add_argument("--histo")
    parser.add_argument("--epochs")
    parser.add_argument("--lr")
    parser.add_argument("--preload")
    parser.add_argument("--preload_exp")
    parser.add_argument("--b_knoise")
    parser.add_argument("--knoise_std")
    parser.add_argument("--tgv")
    parser.add_argument("--metab7T")
    parser.add_argument("--water7T")
    parser.add_argument("--metab3T")
    parser.add_argument("--water3T")
    parser.add_argument("--ntimepts")
    parser.add_argument("--ntimepts3T")
    parser.add_argument("--oversampleTime")
    
    args = parser.parse_args()

    if args.name:
        params["model_name"] = args.name
        params["path_to_model"] = params["path"] + "reco/models/" + params["model_name"] + "/"
        params["path_to_predictions"] = params["path_to_model"] + "predictions/"
        params["path_to_writer"] = params["path_to_model"]
    if args.cc:
        params["use_cc"] = str2bool(args.cc)
        if params["use_cc"]:
            params["num_features_img"] = 2
        else:
            params["num_features_img"] = 64
    if args.undersample_time:
        params["b_undersampleTime"] = str2bool(args.undersample_time)
    if args.aug:
        params["b_aug"] = str2bool(args.aug)
    if args.aug_phase:
        params["b_globPhase"] = str2bool(args.aug_phase)
        params["b_coilPhase"] = str2bool(args.aug_phase)
    if args.aug_noise:
        params["b_noise"] = str2bool(args.aug_noise)
    if args.aug_coilPerm:
        params["b_coilPerm"] = str2bool(args.aug_coilPerm)
    if args.b0_corr:
        params["use_B0_corrections"] = str2bool(args.b0_corr)
    if args.lowrank:
        params["use_lowrank"] = str2bool(args.lowrank)
    if args.acc_low:
        params["acc_low"] = int(args.acc_low)
    if args.acc_high:
        params["acc_high"] = int(args.acc_high)
    if args.loss_space:
        params["loss_space"] = args.loss_space
    if args.loss_tag:
        params["loss_tag"] = args.loss_tag
    if args.epochs:
        params["epochs"] = int(args.epochs)
    if args.lr:
        params["lr"] = float(args.lr)
    if args.preload:
        params["load_model"] = str2bool(args.preload)
    if args.preload_exp:
        params["load_exp_name"] = args.preload_exp
    if args.b_knoise:
        params["b_knoise"] = str2bool(args.b_knoise)
    if args.knoise_std:
        params["knoise_std"] = float(args.knoise_std)
    if args.tgv:
        params["use_tgv"] = str2bool(args.tgv)
    if args.water7T:
        if str2bool(args.water7T):
            params['train_paths'] +=  []

    if args.metab7T:
        if str2bool(args.metab7T):
            
            params['train_paths'] +=[] 
    
    if args.metab3T:
        if str2bool(args.metab3T):
            params['train_paths'] += []
            
    if args.water3T:
        if str2bool(args.water3T):
            params['train_paths'] += []
    
    if args.ntimepts: # valid only for MRSI
        params["ntimepts"] = int(args.ntimepts)
    if args.ntimepts3T: # valid only for MRSI
        params["ntimepts3T"] = int(args.ntimepts3T)
    if args.oversampleTime:
        params["b_oversampleTime"] = str2bool(args.oversampleTime)
    
    return params


def intialize_model_folder(params):
    if params["b_clean_test"] == True:
        if params["model_name"] != "test":
            if os.path.isdir(params["path_to_model"]) == True:
                print("Model already exists. Choose different model name.")
                sys.exit()
            else:
                #os.makedirs(params["path_to_model"])
                os.makedirs(params["path_to_predictions"])
                #my_copy(params["path_to_model"])
        else:
            if os.path.isdir(params["path_to_model"]) == True:
                #shutil.rmtree(params["path_to_model"]+"logs")
                shutil.rmtree(params["path_to_model"])
            #os.makedirs(params["path_to_model"])
            os.makedirs(params["path_to_predictions"])
    my_copy(params["path_to_model"])


def my_copy(path_to_model):
    path = "src/"
    now = datetime.now()
    dest = path_to_model + "src " + str(now) +"/"

    os.makedirs(dest)
    #shutil.copy("srun.sh", dest + "srun.sh")
    shutil.copy("run.py", dest + "run.py")
    shutil.copy("config.py", dest + "config.py")

    dest = dest+"src/"
    os.mkdir(dest)
    src_files = os.listdir(path)
    for file_name in src_files:
        full_file_name = os.path.join(path, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest + file_name)

def save_params(params):

    cmd = " ".join(sys.argv)
    now = datetime.now()
    f = open(params["path_to_model"] + "params.txt", "a")
    f.write("+++ EXECUTION: " + str(now) + " +++")
    f.write('\n')
    f.write("CMD: python " + cmd)
    f.write('\n')
    for key in list(params.keys()):
        f.write(key)
        f.write(": ")
        f.write(str(params[key]))
        f.write('\n')
    f.write('\n')
    f.write('\n')
    f.write('\n')
    f.close() 


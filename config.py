

params={}
## Model Path ##
params["model_name"] =  "test"
params["path"] = "/autofs/space/somes_002/users/pweiser/"
params["path_to_model"] = params["path"] + "ForHauke/models/" + params["model_name"] + "/"
params["path_to_predictions"] = params["path_to_model"] + "predictions/"
params["path_to_writer"] = params["path_to_model"]

## Training ##
params["epochs"] = 1000
params["n_batches"] = -1
params["n_batches_val"] = -1
params["batch_size"] = 1#5
params["workers"] = 10
params["save_model"] = True
params["save_epochs"] = []
params["verbose"] =True

## Optimization ##
params["OPTIONS_loss_tag"] = ["mse", "ncc", "ssim", "ssimse", "001ssimse"]
params["loss_tag"] = params["OPTIONS_loss_tag"][0]
params["OPTIONS_loss_space"] = ["img", "log_img", "img_rel", "img_vrel", "k", "both"]
params["loss_space"] = params["OPTIONS_loss_space"][0]
params["optimizer_tag"] = "adam"
params["lr"] = 2e-3

## Data ##
params['path_to_data'] = params["path"] + 'ForHauke/datasets/'

params['train_paths'] = ['TestSubMRSI.h5', 'TestSubWATER.h5'] #, 'TestSubMRSI.h5', 'TestSubWATER.h5'


#params['val_water7T_paths'] = []
params['val_water7T_paths'] = ['TestSubWATER.h5'] # 'TestSubWATER.h5'

params['val_water3T_paths'] = []


params['val_mrsi7T_paths'] = ["TestSubMRSI.h5"] # "TestSubMRSI.h5"

params['val_mrsi3T_paths'] = []

params['test_mrsi7T_paths'] = []

params['imgsz'] = (64,64,31)


### Options ###
params["gpu"] = 0
params["b_clean_test"] = True
params["b_train"] = True
params["b_test"] = True

## Load Model ###
params["load_model"] = False
params["load_exp_name"] = "EXP_64"
params["load_model_name"] = "model_last.pt"


### Sampling Options
params["b_oversampleTime"] = False
params["b_undersampleTime"] = False
params["ntimepts"] = None # only for MRSI
params["ntimepts3T"] = None # only for MRSI
params["thisTimepts"]=[]
params["acc_low"] = 1
params["acc_high"] = 1#7
params["mrsi_acc_low"] = 2#2
params["mrsi_acc_high"] = 2#6


### Pipline Options ###
params["use_hamming"] = False
params["use_homCorr"] = False
params["use_B0_corrections"] = True
params["use_cc"] = False
if params["use_cc"]:
    params["num_features_img"] = 2
else:
    params["num_features_img"] = 64
params["use_tgv"] =True
params["num_features_img_inter"] = 64
params["OPTIONS_norm"] = ["BatchNorm", "InstanceNorm", "None"]
params["use_norm"] = params["OPTIONS_norm"][1]

### Augmentations ###
params["b_globPhase"] = True
params["b_coilPhase"] = False
params["b_coilPerm"] = False
params["b_aug"] = False
params["b_noise"] = False
params["noise_std"] = 0.02
params["b_knoise"] = False
params["knoise_std"] = 0.002



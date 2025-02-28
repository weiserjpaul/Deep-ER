import numpy as np
import torch
import pandas as pd
import h5py
import nibabel as nib
import random
import time
import monai
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.training_tools import h5group_to_dict, get_timepoints
from src.mr_tools import torch_reco_img_to_kspace, torch_reco_kspace_to_img, reco_nonCart_to_Img_torch, SamplingDensityVoronoi, create_brute_force_FT_matrix_torch

def unfoldBatch(batch):
    all_x = []
    for x in batch:
        if type(x) is dict or type(x) is list:
            #print('dict/list')
            all_x.append(x)
        else:
            #print(x.dtype)
            all_x.append(torch.squeeze(x, dim=0))
    return all_x



def comp_gridded_ham(imgsz):
    x,y,z=imgsz
    xhalf, yhalf, zhalf = int(x/2), int(y/2), int(z/2)
    xx, yy, zz = np.meshgrid(np.arange(x), np.arange(y), np.arange(z))
    dist = np.linalg.norm(np.stack((xx-xhalf, yy-yhalf, zz-zhalf), axis=1), axis=1)
    max_dist = np.max(dist) * 1.1
    ham=1
    hamming_grid = (1-ham/2)+ham/2*np.cos(np.pi*dist/max_dist)
    return hamming_grid

class WaterBatchSamplerNonCart():
    def __init__(self, path_to_data, files, batch_size, undersampleTime, shuffle, 
                    thisTimepts=None, ntimepts=None, ntimepts3T=None, oversampleTime=False):
        self.path_to_data=path_to_data
        self.files=files
        self.batch_size=batch_size
        self.shuffle=shuffle#False#shuffle
        self.undersampleTime=undersampleTime
        self.all_batch=[]
        self.ntimepts=ntimepts
        self.ntimepts3T=ntimepts3T
        self.oversampleTime=oversampleTime
        self.thisTimepts=thisTimepts
        self.sample_indices()
        

    def sample_indices(self):
        self.all_batch=[]
        for f in self.files:
            hf = h5py.File(self.path_to_data + f, 'r')
            if 'MRSI' in f:
                if '3T' in f and self.ntimepts3T:
                    timepts = self.ntimepts3T
                elif self.ntimepts:
                    timepts = self.ntimepts
                else:
                    timepts = get_timepoints(hf)
            else:
                timepts = get_timepoints(hf)
            
            if self.undersampleTime:
                all_timepts = self.undersample_indices(timepts)
            elif self.thisTimepts:
                all_timepts=[self.thisTimepts]
            else: 
                all_timepts = list(range(timepts))
            if self.oversampleTime:
                for i in range(5):
                    new_timepts = list(range(10*(i+1)))
                    all_timepts = all_timepts + new_timepts
            if self.shuffle:
                random.shuffle(all_timepts)
            count=1
            batch=[]
            for i in all_timepts:
                if count%self.batch_size!=0:
                    batch.append((self.path_to_data + f,i))
                    count+=1
                else:
                    batch.append((self.path_to_data + f,i))
                    self.all_batch.append(batch)
                    batch=[]
                    count=1
            if len(batch)>0:
                self.all_batch.append(batch)
        if self.shuffle:
            random.shuffle(self.all_batch)

    def undersample_indices(self, timepts):
        all_timepts = []
        k = -2/3
        d = 7/6
        for i in range(timepts):
            if i/timepts < 0.25:
                all_timepts.append(i)
            else:
                prob = i/timepts * k + d
                if np.random.binomial(1, prob):
                    all_timepts.append(i)
        return all_timepts
        
    def __iter__(self):
        
        self.sample_indices()
        for batch in self.all_batch:
            yield batch
    
    def __len__(self):
        return len(self.all_batch)

def load_data(path_to_data, paths):
    dft_dict = {}
    for p in paths:
        hf = h5py.File(path_to_data + p, 'r')
        dft_inv_kz = h5group_to_dict(hf['ifft_nonCart'])
        dft_dict[path_to_data + p] = dft_inv_kz
    return dft_dict


class WaterBatchDatasetNonCart(Dataset):
    def __init__(self, imgsz, params, acc="random", seed=False):
        self.imgsz=imgsz
        self.use_hamming=params["use_hamming"]
        self.use_homCorr=params["use_homCorr"]
        self.acc=acc
        self.seed=seed
        if self.use_hamming:
            self.hamming_grid=torch.tensor(comp_gridded_ham(imgsz), dtype=torch.float32)
        else:
            self.hamming_grid=False
        self.homCorr_grid=False
        self.acc_low = params["acc_low"]
        self.acc_high = params["acc_high"]
        self.mrsi_acc_low = params["mrsi_acc_low"]
        self.mrsi_acc_high = params["mrsi_acc_high"]
        self.tgv = params["use_tgv"]
    
    def __len__(self):
        return None
        
    def __getitem__(self, indices):
        path = indices[0][0]
        #print("indices: ", indices)
        tindex=[index[1] for index in indices]
        tindex.sort()
        
        
        ######################################
        #### Load ####
        ######################################
        hf = h5py.File(path, 'r')
        mrsi = "MRSI" in path
        sta_time = time.time()
        if mrsi:
            if self.acc=="random":
                acc = random.randint(self.mrsi_acc_low, self.mrsi_acc_high)
                #acc = self.mrsi_acc_low
                if acc != 2:
                    if '3T' in path:
                        acc_ind = random.randint(0,4)
                    else:
                        #acc_ind = random.randint(0,9) # Commented out for simplification reasons. Uncomment for random undersampling.
                        acc_ind = 0
                else:
                    acc_ind = 0
            else:
                acc = self.acc
                acc_ind = 0
            if "tgv" in hf.keys():
                img_mrsi = torch.tensor(hf["tgv"][:,:,:,tindex], dtype=torch.cfloat)
            elif "img_lr_rrrt_2_0" in hf.keys():
                print("WARNING: TGV data not available. Loading ACC 2 instead")
                img_mrsi = torch.tensor(hf["img_lr_rrrt_2_0"][:,:,:,tindex], dtype=torch.cfloat)
            elif "img_lr_rrrt_1_0" in hf.keys():
                print("WARNING: TGV data not available. Loading ACC 1 instead")
                img_mrsi = torch.tensor(hf["img_lr_rrrt_1_0"][:,:,:,tindex], dtype=torch.cfloat)
            else:
                print('No ground truth data found.')
                sys.exit()
            img_mrsi_acc = torch.tensor(hf["img_lr_rrrt_"+str(acc)+"_"+str(acc_ind)][:,:,:,tindex], dtype=torch.cfloat)

            batchsz = img_mrsi.shape[-1]
            img_mrsi = torch.moveaxis(img_mrsi, -1,0) # move batchsz to first position
            img_mrsi_acc = torch.moveaxis(img_mrsi_acc, -1,0) # move batchsz to first position
            if False:
                maxval=np.array(hf['max'])
                img_mrsi = img_mrsi/maxval
                img_mrsi_acc = img_mrsi_acc/maxval
            mrsiData_nonCart = (img_mrsi, img_mrsi_acc)
            img_tgv = 0
            if "OODBrainMask4" in hf.keys():
                brainMask = torch.tensor(np.array(hf["OODBrainMask4"]), dtype=torch.float32)
            else:
                print("WARNING: Loading weaker OOD-mask")
                brainMask = torch.tensor(np.array(hf["OODBrainMask"]), dtype=torch.float32)
        else:
            sta_time_chunck = time.time()
            mrsiData_nonCart = torch.tensor(hf["mrsiData_nonCart_tkc"][tindex,:,:], dtype=torch.cfloat)
            sto_time_chunck=time.time()-sta_time_chunck
            
            batchsz = mrsiData_nonCart.shape[0]
            brainMask = torch.tensor(np.array(hf["BrainMask"]), dtype=torch.float32)
           
            if self.tgv:
                img_tgv = torch.tensor(hf["tgv"][:,:,:,tindex], dtype=torch.cfloat)
                img_tgv = torch.moveaxis(img_tgv, -1, 0)
            else:
                img_tgv = False
        sto_time=time.time()-sta_time
        #print("load time 1: ", sto_time)

        homCorr = torch.tensor(np.array(hf["homCorr"]), dtype=torch.float32)
        sense = torch.tensor(np.array(hf["sense"]))
        if "sense_extrapolate" in hf.keys():
            sense_ext = torch.tensor(np.array(hf["sense_extrapolate"]))
        else:
            print("WARNING: Extrapolated sensitivity mask not found.")
            sense_ext = torch.tensor(np.array(hf["sense"]))
        hamming = torch.tensor(hf['hamming_nonCart_all2'], dtype=torch.float32)
        
        traj = torch.tensor(np.array(hf['trajectories']))
        tindex = torch.tensor(tindex, dtype=torch.float32)

        if not mrsi:
            voronoi = h5group_to_dict(hf["ACC_1+"]["0"])
            voronoi_acc = []
            for i in range(batchsz):
                if self.acc=="random":
                    acc = random.randint(self.acc_low, self.acc_high)
                    acc_str = str(acc)
                    if acc<=5:
                        acc_str += "+"
                else:
                    acc = self.acc
                    acc_str = str(self.acc)
                if self.seed==False and acc>1:
                    #sample = random.randint(0,99) # Commented out for simplification reasons. Uncomment for random undersampling.
                    sample = 0
                else:
                    sample = 0
                key1="ACC_"+acc_str
                key2=str(sample)
                voronoi_acc.append(h5group_to_dict(hf[key1][key2]))
        else:
            voronoi = 0
            voronoi_acc = []
        
        kz_min=-15
        kz_max=15
        Ind_kz = {}
        hamming_kz = {}
        GridX={}
        GridY={}
        for k_z in range(kz_min,kz_max+1):
            ind_z = (k_z%31)
            Ind_kz[str(ind_z)]=np.where(k_z==traj[:,2])[0]
            GridX[str(ind_z)]=traj[Ind_kz[str(ind_z)],0]
            GridY[str(ind_z)]=traj[Ind_kz[str(ind_z)],1]
            if not mrsi:
                voronoi[str(ind_z)] = torch.tensor(voronoi[str(ind_z)], dtype=torch.cfloat)
                for v in voronoi_acc:
                    v[str(ind_z)] = torch.tensor(v[str(ind_z)], dtype=torch.cfloat)
            hamming_kz[str(ind_z)] = hamming[Ind_kz[str(ind_z)]]
        
        

        return mrsiData_nonCart, img_tgv, Ind_kz, voronoi, voronoi_acc, GridX, GridY, hamming_kz, homCorr, sense, sense_ext, \
                self.hamming_grid, self.homCorr_grid, tindex, mrsi, brainMask, sto_time


def undersampleNonCart(NbAng, NbTilt, Ind_kz, kz_min, kz_max, acc, seed):
    if acc=="random":
        acc = random.randint(1, 6)
    UndersampleF = 1/acc
    NbAng2keep = int(np.round(NbAng*UndersampleF))
    if seed:
        np.random.seed(0)
    I = np.argsort(np.random.uniform(size=(NbAng)))
    Ang2keep=np.zeros((NbAng), dtype=np.float32)
    Ang2keep[I[:NbAng2keep]]=1
    Ang2keep = np.repeat(Ang2keep[:,None], repeats=NbTilt, axis=1)
    Ang2keep = np.reshape(Ang2keep, (-1,))

    return torch.tensor(Ang2keep)
    

def preprocessNonCart(mrsiData_nonCart, img_tgv, Ind_kz, voronoi, voronoi_acc, GridX, GridY, 
                        hamming_kz, homCorr, sense, sense_ext, hamming_grid, homCorr_grid,
                        tindex, params, b_coilPhase, b_globPhase, b_noise, 
                        b_knoise, b_aug, b_coilPerm, b_tgv, mrsi, brainMask, device):
    

    kz_min=-15
    kz_max=15
    ######################################
    ### To Device ###
    ######################################

    dft_inv_kz = {}

    for k_z in range(kz_min,kz_max+1):
        ind_z = (k_z%31)
        hamming_kz[str(ind_z)] = hamming_kz[str(ind_z)][0].to(device)
        dft_kz = create_brute_force_FT_matrix_torch([64,64], GridX[str(ind_z)][0].to(device), GridY[str(ind_z)][0].to(device), device)
        dft_inv_kz[str(ind_z)] = torch.transpose(torch.conj(dft_kz),1,0)
        if not mrsi:
            voronoi[str(ind_z)] = voronoi[str(ind_z)][0].to(device)
            for v in voronoi_acc:
                v[str(ind_z)] = v[str(ind_z)][0].to(device)
    if not mrsi:
        mrsiData_nonCart = mrsiData_nonCart.to(device)
    else:
        mrsiData_nonCart = (mrsiData_nonCart[0][0].to(device), mrsiData_nonCart[1][0].to(device))
    sense = sense.to(device)
    sense_ext = sense_ext.to(device)
    homCorr = homCorr.to(device)
    tindex = tindex.to(device)
    img_tgv = img_tgv.to(device)
    brainMask = brainMask.to(device)
    if not hamming_grid is None:
        hamming_grid = hamming_grid.to(device)

    ######################################
    ### Augment in k-space Space ###
    ######################################

    
    if not mrsi:
        batchsz = mrsiData_nonCart.shape[0]
        coils = mrsiData_nonCart.shape[-1]
    else:
        batchsz = mrsiData_nonCart[0].shape[0]
        coils = sense.shape[-1]
        
        
    if b_coilPhase:
        phase_coil = torch.exp(2j*np.pi*torch.rand(batchsz, coils)).to(device)
    else:
        phase_coil = torch.ones(batchsz, coils).to(device)
    if b_globPhase:
        phase_glob = torch.exp(2j*np.pi*torch.rand(batchsz)).to(device)
    else:
        phase_glob = torch.ones(batchsz).to(device)
    
    if not mrsi:
        mrsiData_nonCart = mrsiData_nonCart*phase_glob[:,None,None]*phase_coil[:,None,:]
        if b_tgv:
            img_tgv = img_tgv*phase_glob[:,None,None,None]
    else:
        mrsiData_nonCart = (mrsiData_nonCart[0]*phase_glob[:,None,None,None], mrsiData_nonCart[1]*phase_glob[:,None,None,None])

    if b_coilPerm and not mrsi:
        perm = torch.zeros((batchsz, coils), dtype=torch.long).to(device)
        for i in range(batchsz):
            perm[i] = torch.randperm(coils)
            mrsiData_nonCart[i] = mrsiData_nonCart[i,:,perm[i]]
            phase_coil[i] = phase_coil[i,perm[i]]
    else:
        perm = torch.tensor(range(coils))
        perm = perm[None,].repeat(batchsz,1)
    
    ######################################
    ### nonCart k-space to Image Space ###
    ######################################
    
    if not mrsi:
        for i in range(batchsz):
            if b_knoise:
                curr_knoise_std = (torch.rand(1) * params["knoise_std"]).to(device)

            mrsiData_kz = {}
            mrsiData_kz_under = {}
            for k_z in range(kz_min,kz_max+1):
                ind_z = (k_z%31)
                if b_knoise:
                    s = mrsiData_nonCart[i,Ind_kz[str(ind_z)],:].shape
                    noise = curr_knoise_std * (torch.randn(s, device=device) + 1j*torch.randn(s, device=device))
                    mrsiData_kz[str(ind_z)] = mrsiData_nonCart[i,Ind_kz[str(ind_z)],:] + noise
                else:
                    mrsiData_kz[str(ind_z)] = mrsiData_nonCart[i,Ind_kz[str(ind_z)],:]
            
            if i == 0:
                img_cc = reco_nonCart_to_Img_torch(mrsiData_kz=mrsiData_kz, 
                                            hamming_kz=hamming_kz, 
                                            dft_inv_kz=dft_inv_kz, 
                                            voronoi=voronoi,
                                            sense=sense[:,:,:,perm[i]]*phase_coil[i,None,None,None,:], 
                                            homCorr=homCorr,
                                            device = device,
                                            coils=32,
                                            use_cc=params["use_cc"],
                                            mrsi=mrsi)[None,:]
                img_cc_under = reco_nonCart_to_Img_torch(mrsiData_kz=mrsiData_kz, 
                                                        hamming_kz=hamming_kz, 
                                                        dft_inv_kz=dft_inv_kz, 
                                                        voronoi=voronoi_acc[i],
                                                        sense=sense[:,:,:,perm[i]]*phase_coil[i,None,None,None,:], 
                                                        homCorr=homCorr,
                                                        device = device,
                                                        coils=32,
                                                        use_cc=params["use_cc"],
                                                        mrsi=mrsi)[None,:]
            else:
                img_cc = torch.cat((img_cc, reco_nonCart_to_Img_torch(mrsiData_kz=mrsiData_kz, 
                                                                    hamming_kz=hamming_kz, 
                                                                    dft_inv_kz=dft_inv_kz, 
                                                                    voronoi=voronoi,
                                                                    sense=sense[:,:,:,perm[i]]*phase_coil[i,None,None,None,:], 
                                                                    homCorr=homCorr,
                                                                    device = device,
                                                                    coils=32,
                                                                    use_cc=params["use_cc"],
                                                                    mrsi=mrsi)[None,:]
                                    ), dim=0
                                    )
                img_cc_under = torch.cat((img_cc_under, reco_nonCart_to_Img_torch(mrsiData_kz=mrsiData_kz, 
                                                                                hamming_kz=hamming_kz, 
                                                                                dft_inv_kz=dft_inv_kz, 
                                                                                voronoi=voronoi_acc[i],
                                                                                sense=sense[:,:,:,perm[i]]*phase_coil[i,None,None,None,:], 
                                                                                homCorr=homCorr,
                                                                                device = device,
                                                                                coils=32,
                                                                                use_cc=params["use_cc"],
                                                                                mrsi=mrsi)[None,:]
                                    ), dim=0
                                    )
    
    
    if mrsi:
        img_cc = mrsiData_nonCart[0] * brainMask[None,]
        img_cc_under = mrsiData_nonCart[1] * brainMask[None,]
        
        if params["use_cc"] == False:
            img_cc = torch.unsqueeze(img_cc, dim=-1) * sense[None,:]
            img_cc = torch.moveaxis(img_cc, -1, 1)
            img_cc_under = torch.unsqueeze(img_cc_under, dim=-1) * sense[None,:] 
            img_cc_under = torch.moveaxis(img_cc_under, -1, 1)
    if b_tgv and not mrsi:
        img_cc = img_tgv
        if params["use_cc"] == False:
            img_cc * homCorr[None,]
            img_cc = torch.unsqueeze(img_cc, dim=-1) * (sense[None,:,:,:,perm[i]] * phase_coil[:,None,None,None,:])
            img_cc = torch.moveaxis(img_cc, -1, 1)
    if not mrsi:
        if True:
            
            headmask=torch.abs(sense[:,:,:,0])>0
            img_cc = img_cc * headmask[None,None] #* brainMask[None,]
            img_cc_under = img_cc_under * headmask[None,None] #* brainMask[None,]

    ######################################
    ### Augment in Image Space ###
    ######################################
    
    if True:
        if params["use_cc"]:
            if mrsi:
                img_max = []
                for i in range(batchsz):
                    tmp = torch.abs(img_cc_under[i])
                    img_max.append(torch.quantile(torch.reshape(tmp[tmp>0], (-1,)), 0.95))
                img_max = torch.tensor(img_max).to(device)
                img_cc = img_cc/img_max[:,None,None,None]
                img_cc_under = img_cc_under/img_max[:,None,None,None]
            else:
                img_max = torch.amax(torch.abs(img_cc_under), dim=(1,2,3))
                img_cc = img_cc/img_max[:,None,None,None]
                img_cc_under = img_cc_under/img_max[:,None,None,None]
        else:
            img_max = []
            for i in range(batchsz):
                tmp = torch.abs(img_cc_under[i])
                img_max.append(torch.quantile(torch.reshape(tmp[tmp>0], (-1,)), 0.98))
            img_max = torch.tensor(img_max).to(device)
            img_cc = img_cc/img_max[:,None,None,None,None]
            img_cc_under = img_cc_under/img_max[:,None,None,None,None]
    else:
        img_max=0

   

    if b_noise:
        std = (torch.rand(batchsz) * params["noise_std"]).to(device)
        rand = (torch.randn(tuple(img_cc.shape)) + 1j * torch.randn(tuple(img_cc.shape))).to(device)
        img_cc_under = img_cc_under + (std[(..., ) + (None,)*(len(img_cc.shape)-1)] * rand)
        img_cc = img_cc + (std[(..., ) + (None,)*(len(img_cc.shape)-1)] * rand)

    brainMaskAug = torch.ones((batchsz,)+brainMask.shape).to(device)
    brainMaskAug = brainMaskAug*brainMask[None,:]
    if b_aug:
        
        affine_deform = monai.transforms.Rand3DElastic(sigma_range = (5,8), 
                                           magnitude_range = (200,250), 
                                           prob=1, 
                                           rotate_range=(.3,.3,.3), 
                                           shear_range=(0.1,0.1,0.1,0.1,0.1,0.1), 
                                           translate_range=(6,6,3), 
                                           scale_range=(0.1,0.1,0.1), 
                                           spatial_size=None, 
                                           mode="nearest", 
                                           padding_mode="zeros", 
                                           device=device
                                          )
        
        field_size = random.randint(4,10)
        expo_deform = monai.transforms.RandSmoothFieldAdjustIntensity(spatial_size = (64,64,31), 
                                                                    rand_size = (field_size,field_size,int(field_size/2)), 
                                                                    pad=0, 
                                                                    mode="trilinear", 
                                                                    align_corners=None, 
                                                                    prob=1, 
                                                                    gamma=(0.3, 1.3), 
                                                                    device=device)


        for i in range(batchsz):
            
            if params["use_cc"]:
                x = torch.stack((torch.real(img_cc[i]), torch.imag(img_cc[i])), axis=0)
                y = torch.stack((torch.real(img_cc_under[i]), torch.imag(img_cc_under[i])), axis=0)
            else:
                x = torch.cat((torch.real(img_cc[i]), torch.imag(img_cc[i])), axis=0)
                y = torch.cat((torch.real(img_cc_under[i]), torch.imag(img_cc_under[i])), axis=0)
            
            xy = torch.cat((x,y), dim=0)
            xy = expo_deform(xy)
            xy = torch.cat((brainMask[None,:],xy), dim=0)
            xy = affine_deform(xy)
            
        
            if params["use_cc"]:
                brainMaskAug[i] = xy[0]
                x = xy[1:2+1]
                y = xy[2+1:]
                img_cc[i] = x[0] + 1j*x[1]
                img_cc_under[i] = y[0] + 1j*y[1]
            else:
                brainMaskAug[i] = xy[0]
                x = xy[1:64+1]
                y = xy[64+1:]
                img_cc[i] = x[:32] + 1j*x[32:]
                img_cc_under[i] = y[:32] + 1j*y[32:]
            

    ######################################
    ### Image Space to Gridded k-space ###
    ######################################
    for i in range(batchsz):
        if i == 0:
            k_coils = torch_reco_img_to_kspace(img_cc=img_cc[i], 
                                                sense=sense_ext[:,:,:,perm[i]]*phase_coil[i,None,None,None,:],
                                                hamming_grid=hamming_grid, 
                                                homCorr=homCorr_grid,
                                                use_cc=params["use_cc"])[None,:]
            k_coils_under = torch_reco_img_to_kspace(img_cc=img_cc_under[i], 
                                                    sense=sense_ext[:,:,:,perm[i]]*phase_coil[i,None,None,None,:],
                                                    hamming_grid=hamming_grid, 
                                                    homCorr=homCorr_grid,
                                                    use_cc=params["use_cc"])[None,:]
        else:
            k_coils = torch.cat((k_coils, torch_reco_img_to_kspace(img_cc=img_cc[i], 
                                                                    sense=sense_ext[:,:,:,perm[i]]*phase_coil[i,None,None,None,:],
                                                                    hamming_grid=hamming_grid, 
                                                                    homCorr=homCorr_grid,
                                                                    use_cc=params["use_cc"])[None,:]
                                    ), dim=0
                                )
            k_coils_under = torch.cat((k_coils_under, torch_reco_img_to_kspace(img_cc=img_cc_under[i], 
                                                                                sense=sense_ext[:,:,:,perm[i]]*phase_coil[i,None,None,None,:],
                                                                                hamming_grid=hamming_grid, 
                                                                                homCorr=homCorr_grid,
                                                                                use_cc=params["use_cc"])[None,:]
                                    ), dim=0
                                )
    
    ######################
    ###### reshape #######
    ######################
    if params["use_cc"]:
        img_cc = torch.cat((torch.real(img_cc)[:,None], torch.imag(img_cc)[:,None]), dim=1)
        img_cc_under = torch.cat((torch.real(img_cc_under)[:,None], torch.imag(img_cc_under)[:,None]), dim=1)
    else:
        img_cc = torch.cat((torch.real(img_cc), torch.imag(img_cc)), dim=1)
        img_cc_under = torch.cat((torch.real(img_cc_under), torch.imag(img_cc_under)), dim=1)


    return img_cc_under, k_coils_under, img_cc, k_coils, sense, sense_ext, perm, hamming_grid, homCorr, homCorr_grid, img_max, phase_coil, brainMask, brainMaskAug
    


def postprocessNonCart(reco_kspace, mrsiData_Cart, sense, perm, phase_coil, hamming_grid, homCorr, brainMask, brainMaskAug, params, mrsiData_Cart_under=None):
    batchsz = reco_kspace.shape[0]
        
    for i in range(batchsz):
        if i == 0:
            reco_img = torch.unsqueeze(torch_reco_kspace_to_img(k_coils=reco_kspace[i],
                                                                sense=sense[:,:,:,perm[i]]*phase_coil[i,None,None,None,:], 
                                                                hamming_grid=hamming_grid, 
                                                                homCorr=homCorr,
                                                                brainMask=brainMaskAug[i],
                                                                use_cc=params["use_cc"],
                                                                final=True),
                                        dim=0)
            
            img = torch.unsqueeze(torch_reco_kspace_to_img(k_coils=mrsiData_Cart[i],
                                                            sense=sense[:,:,:,perm[i]]*phase_coil[i,None,None,None,:], 
                                                            hamming_grid=hamming_grid, 
                                                            homCorr=homCorr,
                                                            brainMask=brainMaskAug[i],
                                                            use_cc=params["use_cc"],
                                                            final=True),
                                    dim=0)

            if mrsiData_Cart_under is not None:
                img_under = torch.unsqueeze(torch_reco_kspace_to_img(k_coils=mrsiData_Cart_under[i],
                                                                sense=sense[:,:,:,perm[i]]*phase_coil[i,None,None,None,:], 
                                                                hamming_grid=hamming_grid, 
                                                                homCorr=homCorr,
                                                                brainMask=brainMaskAug[i],
                                                                use_cc=params["use_cc"],
                                                                final=True),
                                        dim=0)
            
        else:
            reco_img = torch.cat((reco_img, torch.unsqueeze(torch_reco_kspace_to_img(k_coils=reco_kspace[i],
                                                                                    sense=sense[:,:,:,perm[i]]*phase_coil[i,None,None,None,:], 
                                                                                    hamming_grid=hamming_grid, 
                                                                                    homCorr=homCorr,
                                                                                    brainMask=brainMaskAug[i],
                                                                                    use_cc=params["use_cc"],
                                                                                    final=True), 
                                                            dim=0)
                                    ), 
                                    dim=0)
            
            img = torch.cat((img, torch.unsqueeze(torch_reco_kspace_to_img(k_coils=mrsiData_Cart[i],
                                                                            sense=sense[:,:,:,perm[i]]*phase_coil[i,None,None,None,:], 
                                                                            hamming_grid=hamming_grid, 
                                                                            homCorr=homCorr,
                                                                            brainMask=brainMaskAug[i],
                                                                            use_cc=params["use_cc"],
                                                                            final=True),
                                                    dim=0)
                                ), 
                                dim=0)
            if mrsiData_Cart_under is not None:
                img_under = torch.cat((img_under, torch.unsqueeze(torch_reco_kspace_to_img(k_coils=mrsiData_Cart_under[i],
                                                                                sense=sense[:,:,:,perm[i]]*phase_coil[i,None,None,None,:], 
                                                                                hamming_grid=hamming_grid, 
                                                                                homCorr=homCorr,
                                                                                brainMask=brainMaskAug[i],
                                                                                use_cc=params["use_cc"],
                                                                                final=True), 
                                                        dim=0)
                                    ), 
                                    dim=0)

    
    if mrsiData_Cart_under is not None:
        return reco_img, img, img_under
    else:
        return reco_img, img
    
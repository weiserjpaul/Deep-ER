import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import random
import h5py
from PIL import Image
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F



from src.mr_tools import torch_reco_img_to_kspace, torch_reco_kspace_to_img
from src. dataloaderBatchNonCart import preprocessNonCart, postprocessNonCart, unfoldBatch





def training(model, train_dataloader, optimizer, params, epoch, device, writer=None):

    model.train()
    #model.eval()
    epoch_loss = []
    sta_epoch = time.time()
    
    if params["n_batches"] == -1 or params["n_batches"] > len(train_dataloader):
        n_batches_tmp = len(train_dataloader)
    else:
        n_batches_tmp = params["n_batches"]

    for i, batch in enumerate(train_dataloader):
        if i >= n_batches_tmp:
            break
        sta_batch = time.time()
        mrsiData_nonCart, img_tgv, Ind_kz, voronoi, voronoi_acc, GridX, GridY, hamming_kz, homCorr, \
            sense, sense_ext, hamming_grid, homCorr_grid, tindex, mrsi, brainMask, sto_time = unfoldBatch(batch)
        
        img_under_v0, mrsiData_Cart_under, imgt, mrsiData_Cart, sense, sense_ext, perm, hamming_grid, homCorr, homCorr_grid, \
            img_max, phase_coil, brainMask, brainMaskAug = preprocessNonCart(mrsiData_nonCart=mrsiData_nonCart, 
                                                                            img_tgv = img_tgv,
                                                                            Ind_kz=Ind_kz, 
                                                                            voronoi=voronoi,
                                                                            voronoi_acc=voronoi_acc,
                                                                            GridX=GridX, 
                                                                            GridY=GridY, 
                                                                            hamming_kz=hamming_kz, 
                                                                            homCorr=homCorr, 
                                                                            sense=sense, 
                                                                            sense_ext=sense_ext,
                                                                            hamming_grid=hamming_grid, 
                                                                            homCorr_grid=homCorr_grid,
                                                                            tindex=tindex, 
                                                                            params=params, 
                                                                            b_coilPhase=params["b_coilPhase"], 
                                                                            b_globPhase=params["b_globPhase"], 
                                                                            b_noise=params["b_noise"],
                                                                            b_knoise=params["b_knoise"],
                                                                            b_aug=params["b_aug"],
                                                                            b_coilPerm=params["b_coilPerm"],
                                                                            b_tgv=params["use_tgv"],
                                                                            mrsi=mrsi,
                                                                            brainMask=brainMask,
                                                                            device=device)

        reco_img1, reco_kspace = model((img_under_v0, mrsiData_Cart_under), 
                                        hamming_grid, 
                                        sense_ext,
                                        perm,
                                        phase_coil,
                                        homCorr_grid
                                        )
        
            
        
        reco_img, img, img_under = postprocessNonCart(reco_kspace=reco_kspace, 
                                                    mrsiData_Cart=mrsiData_Cart,
                                                    mrsiData_Cart_under=mrsiData_Cart_under,
                                                    sense=sense_ext,
                                                    perm=perm,
                                                    phase_coil=phase_coil, 
                                                    hamming_grid=hamming_grid, 
                                                    homCorr=homCorr, 
                                                    brainMask=brainMask,
                                                    brainMaskAug=brainMaskAug,
                                                    params=params)


        

        if False:
            fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(15,3))
            ax[0].imshow(np.abs(img[0,0,:,:,15].cpu() + 1j*img[0,1,:,:,15].cpu()), cmap="gray")
            ax[1].imshow(np.abs(mrsiData_Cart[0,5,:,:,15].cpu()))
            ax[2].imshow(np.abs(img_under[0,0,:,:,15].cpu() + 1j*img_under[0,1,:,:,15].cpu()), cmap="gray")
            ax[3].imshow(np.abs(mrsiData_Cart_under[0,5,:,:,15].cpu()))
            ax[4].imshow(np.abs(reco_img[0,0,:,:,15].detach().cpu() + reco_img[0,1,:,:,15].detach().cpu()), cmap="gray")
            ax[5].imshow(np.abs(reco_kspace[0,5,:,:,15].detach().cpu()))
            plt.show()
        

        if False:
            ### Reconstructed Image After PP ###
            print("### Reco Image After PP ###")
            print(reco_img.shape)
            img_v2=reco_img[:,0] + 1j*reco_img[:,1]
            print(torch.amax(torch.abs(img_v2[0])))
            print(torch.quantile(torch.abs(img_v2[0]),q=0.95))

            vmax=torch.amax(torch.abs(img_v2[0]))
            quant95=torch.quantile(torch.abs(img_v2[0]),q=0.95)+2
            quant97=torch.quantile(torch.abs(img_v2[0]),q=0.97)
            quant99=torch.quantile(torch.abs(img_v2[0]),q=0.99)
            img_v4=torch.clone(img_v2)
            img_v4[0][torch.abs(img_v4[0])<quant95]=float('nan')
            save_img4 = np.array(img_v4[0].cpu().detach())
            save_img = np.array(img_v2[0].cpu().detach())
            
            fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(20,25))
            str_title="Image Space, \n max: " + str(torch.round(vmax, decimals=1).item()) + ", 95th quantile: "+str(torch.round(quant95, decimals=1).item())+"\n"
            str_title=str_title+"97th quantile: "+str(torch.round(quant97, decimals=1).item())+"\n"+", 99th quantile: "+str(torch.round(quant99, decimals=1).item())+"\n"
            fig.suptitle(str_title, fontsize=16)
            for jj in range(6):
                for ii in range(5):
                    sl = ii + jj*5
                    ax[jj,ii].imshow(np.abs(save_img[:,:,sl]), cmap='gray',vmin=0,vmax=quant95)
                    ax[jj,ii].title.set_text('Slice: ' + str(sl))
            fig.tight_layout()
            plt.savefig(params["path_to_predictions"]+"epoch"+str(epoch)+"_batch"+str(i)+"_RECOImgafterPP_m"+str(mrsi.item()))
            plt.close(fig)


        if False:
            ### Fully Sampled Image After PP ###
            print("### Fully Sampled Image After PP ###")
            img_v2=img[:,0] + 1j*img[:,1]
            print(torch.amax(torch.abs(img_v2[0])))
            print(torch.quantile(torch.abs(img_v2[0]),q=0.95))
            save_img = np.array(img_v2[0].cpu().detach())
            
            fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(20,25))
            fig.suptitle("Image Space, \n", fontsize=16)
            for jj in range(6):
                for ii in range(5):
                    sl = ii + jj*5
                    ax[jj,ii].imshow(np.abs(save_img[:,:,sl]), cmap='gray',vmin=0,vmax=quant95)
                    ax[jj,ii].title.set_text('Slice: ' + str(sl))
            fig.tight_layout()
            plt.savefig(params["path_to_predictions"]+"epoch"+str(epoch)+"_batch"+str(i)+"_FSImgafterPP_m"+str(mrsi.item()))
            plt.close(fig)

            if False:
                fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(20,25))
                fig.suptitle("Image Space, \n", fontsize=16)
                for jj in range(6):
                    for ii in range(5):
                        sl = ii + jj*5
                        ax[jj,ii].imshow(np.angle(save_img[:,:,sl]))
                        ax[jj,ii].title.set_text('Slice: ' + str(sl))
                fig.tight_layout()
                plt.savefig(params["path_to_predictions"]+"epoch"+str(epoch)+"_batch"+str(i)+"_FSImgafterPP_angel_m"+str(mrsi.item()))
                plt.close(fig)
            
        if False:
            ### Fully Sampled Image before PP ###
            print(imgt.shape)
            imgt_v2=imgt[:,:32] + 1j*imgt[:,32:]
            save_img = np.array(imgt_v2[0].cpu().detach())
            coil=10
            fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(20,25))
            fig.suptitle('Image Space, \n Coil: '+str(coil)+"\n", fontsize=16)
            for jj in range(6):
                for ii in range(5):
                    sl = ii + jj*5
                    ax[jj,ii].imshow(np.abs(save_img[coil,:,:,sl]), cmap='gray')
                    ax[jj,ii].title.set_text('Slice: ' + str(sl))
            fig.tight_layout()
            plt.savefig(params["path_to_predictions"]+"epoch"+str(epoch)+"_batch"+str(i)+"_FSImgbeforePP_m"+str(mrsi.item()))
            plt.close(fig)

            fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(20,25))
            fig.suptitle('Image Space, \n Coil: '+str(coil)+"\n", fontsize=16)
            for jj in range(6):
                for ii in range(5):
                    sl = ii + jj*5
                    ax[jj,ii].imshow(np.angle(save_img[coil,:,:,sl]))
                    ax[jj,ii].title.set_text('Slice: ' + str(sl))
            fig.tight_layout()
            plt.savefig(params["path_to_predictions"]+"epoch"+str(epoch)+"_batch"+str(i)+"_FSImgbeforePP_angel_m"+str(mrsi.item()))
            plt.close(fig)
            

        
            ### Undersampled Image before PP ###
            print(img_under_v0.shape)
            img_under_v2=img_under_v0[:,:32] + 1j*img_under_v0[:,32:]
            save_img = np.array(img_under_v2[0].cpu().detach())
            coil=10
            fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(20,25))
            fig.suptitle('Image Space, \n Coil: '+str(coil)+"\n", fontsize=16)
            for jj in range(6):
                for ii in range(5):
                    sl = ii + jj*5
                    ax[jj,ii].imshow(np.abs(save_img[coil,:,:,sl]), cmap='gray')
                    ax[jj,ii].title.set_text('Slice: ' + str(sl))
            fig.tight_layout()
            plt.savefig(params["path_to_predictions"]+"epoch"+str(epoch)+"_batch"+str(i)+"_USImgbeforePP_m"+str(mrsi.item()))
            plt.close(fig)
            
            fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(20,25))
            fig.suptitle('Image Space, \n Coil: '+str(coil)+"\n", fontsize=16)
            for jj in range(6):
                for ii in range(5):
                    sl = ii + jj*5
                    ax[jj,ii].imshow(np.angle(save_img[coil,:,:,sl]))
                    ax[jj,ii].title.set_text('Slice: ' + str(sl))
            fig.tight_layout()
            plt.savefig(params["path_to_predictions"]+"epoch"+str(epoch)+"_batch"+str(i)+"_USImgbeforePP_angel_m"+str(mrsi.item()))
            plt.close(fig)

        if False:
            ### Undersampled Image after PP ###
            print("### Undersampled Image after PP ###")
            img_v2=img_under[:,0] + 1j*img_under[:,1]

            vmax=torch.amax(torch.abs(img_v2[0]))
            print(torch.amax(torch.abs(img_v2[0])))
            print(torch.quantile(torch.abs(img_v2[0]),q=0.95))
            img_v4=torch.clone(img_v2)
            img_v4[0][torch.abs(img_v4[0])<quant95]=float('nan')
            save_img4 = np.array(img_v4[0].cpu().detach())
            save_img = np.array(img_v2[0].cpu().detach())
            
            fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(20,25))
            fig.suptitle("Undersampled Image Space, \n max: " + str(torch.round(vmax, decimals=1).item()) + ", 95th quantile: "+str(torch.round(quant95, decimals=1).item())+"\n", fontsize=16)
            for jj in range(6):
                for ii in range(5):
                    sl = ii + jj*5
                    ax[jj,ii].imshow(np.abs(save_img[:,:,sl]), cmap='gray',vmin=0,vmax=quant95)
                    #ax[jj,ii].imshow(np.abs(save_img4[:,:,sl]), cmap='Reds', alpha=0.5,vmin=0,vmax=vmax)
                    ax[jj,ii].title.set_text('Slice: ' + str(sl))
            fig.tight_layout()
            plt.savefig(params["path_to_predictions"]+"epoch"+str(epoch)+"_batch"+str(i)+"_USImgafterPP_m"+str(mrsi.item()))
            plt.close(fig)



        if False:
            ### Undersampled k-Space ###
            print(mrsiData_Cart_under.shape)
            mrsiData_Cart_v2 = mrsiData_Cart_under[:,:32] + 1j*mrsiData_Cart_under[:,32:]
            save_img = np.array(mrsiData_Cart_v2[0].cpu().detach())
            coil = 10
            vmax=np.amax(np.abs(save_img[coil,:,:,:]))
            vmin=np.amin(np.abs(save_img[coil,:,:,:]))
            fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(20,25))
            fig.suptitle('Image Space, \n Coil: '+str(coil)+"\n", fontsize=16)
            for jj in range(6):
                for ii in range(5):
                    sl = ii + jj*5
                    ax[jj,ii].imshow(np.abs(save_img[coil,:,:,sl]))#, cmap='gray', vmax=vmax, vmin=vmin)
                    ax[jj,ii].title.set_text('Slice: ' + str(sl))
            fig.tight_layout()
            plt.savefig(params["path_to_predictions"]+"epoch"+str(epoch)+"_batch"+str(i)+"_USKSpace_m"+str(mrsi.item()))
            plt.close(fig)

            fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(20,25))
            fig.suptitle('Image Space, \n Coil: '+str(coil)+"\n", fontsize=16)
            for jj in range(6):
                for ii in range(5):
                    sl = ii + jj*5
                    ax[jj,ii].imshow(np.angle(save_img[coil,:,:,sl]))
                    ax[jj,ii].title.set_text('Slice: ' + str(sl))
            fig.tight_layout()
            plt.savefig(params["path_to_predictions"]+"epoch"+str(epoch)+"_batch"+str(i)+"_USKSpace_angel_m"+str(mrsi.item()))
            plt.close(fig)
        

        
        loss = params["loss_func"](reco_img=reco_img,
                                    img=img,
                                    img_under=img_under,
                                    reco_kspace=reco_kspace,
                                    mrsiData_Cart=mrsiData_Cart,
                                    mrsiData_Cart_under=mrsiData_Cart_under,
                                    mrsi=mrsi,
                                    mask=brainMaskAug)
        
        loss.backward()
        if False:
            print("weight: ", model.conv1d_kspace.weight.shape)
            print("grad IL1: ", torch.mean(model.interlacer_layers[0].img_bnconvs[0].conv.weight.grad))
            print("grad IL1: ", torch.mean(model.interlacer_layers[0].freq_bnconvs[0].conv.weight.grad))
            print("grad IL5: ", torch.mean(model.interlacer_layers[4].img_bnconvs[0].conv.weight.grad))
            print("grad IL5: ", torch.mean(model.interlacer_layers[4].freq_bnconvs[0].conv.weight.grad))
            print("grad IL9: ", torch.mean(model.interlacer_layers[8].img_bnconvs[0].conv.weight.grad))
            print("grad IL9: ", torch.mean(model.interlacer_layers[8].freq_bnconvs[0].conv.weight.grad))
            print("grad: ", torch.mean(model.conv1d_kspace.weight.grad))

        optimizer.step()
        optimizer.zero_grad()
        epoch_loss.append(loss.item())
        
        sto_batch = time.time() - sta_batch
        if params["verbose"]:
            log_batch = ' ~ Epoch: {:03d}, Batch: ({:03d}/{:03d}) Loss: {:.8f}, GPU-Time: {:.4f}, Load-Time: {:.4f}'
            print(log_batch.format(epoch+1, i+1, n_batches_tmp, loss.item(), sto_batch, sto_time))


    epoch_loss = np.mean(np.array(epoch_loss))
    sto_epoch = time.time() - sta_epoch
    log_epoch = 'Epoch: {:03d}, Loss: {:.8f}, Time: {:.4f}'
    print(log_epoch.format(epoch+1, epoch_loss , sto_epoch))
    if writer:
        writer.add_scalars(params["model_name"] + "/loss", {"train": epoch_loss}, epoch)
        writer.flush()
    
    del img_under_v0, mrsiData_Cart_under, imgt, mrsiData_Cart, sense, sense_ext, reco_img1, reco_kspace, reco_img, img, img_under, loss
    torch.cuda.empty_cache()
    
    return model


def validation(model, validation_dataloader, params, epoch, device, writer=None, tag=""):

    model.eval()
    if not tag == "":
        tag = " " + tag 
    with torch.no_grad():

        val_epoch_loss = []
        sta_epoch = time.time()

        if params["n_batches_val"] == -1 or params["n_batches_val"] > len(validation_dataloader):
            n_batches_val_tmp = len(validation_dataloader)
        else: 
            n_batches_val_tmp = params["n_batches_val"]

        for i, batch in enumerate(validation_dataloader):
            if i >= n_batches_val_tmp:
                break
            sta_batch = time.time()

            mrsiData_nonCart, img_tgv, Ind_kz, voronoi, voronoi_acc, GridX, GridY, hamming_kz, homCorr, \
                sense, sense_ext, hamming_grid, homCorr_grid, tindex, mrsi, brainMask, sto_time = unfoldBatch(batch)

            img_under, mrsiData_Cart_under, img, mrsiData_Cart, sense, sense_ext, perm, hamming_grid, homCorr, homCorr_grid, \
                img_max, phase_coil, brainMask,brainMaskAug = preprocessNonCart(mrsiData_nonCart=mrsiData_nonCart, 
                                                                        img_tgv=img_tgv,
                                                                        Ind_kz=Ind_kz, 
                                                                        voronoi=voronoi,
                                                                        voronoi_acc=voronoi_acc,
                                                                        GridX=GridX, 
                                                                        GridY=GridY, 
                                                                        hamming_kz=hamming_kz, 
                                                                        homCorr=homCorr, 
                                                                        sense=sense, 
                                                                        sense_ext=sense, 
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
            
            reco_img1, reco_kspace = model((img_under, mrsiData_Cart_under), 
                                            hamming_grid, 
                                            sense, 
                                            perm,
                                            phase_coil,
                                            homCorr_grid
                                            )
            
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
                ### Reconstructed Image After PP ###
                print(reco_img.shape)
                img_v2=reco_img[:,0] + 1j*reco_img[:,1]
                print(torch.amax(torch.abs(img_v2[0])))
                save_img = np.array(img_v2[0].cpu().detach())
                fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(20,25))
                fig.suptitle("Image Space, \n", fontsize=16)
                for jj in range(6):
                    for ii in range(5):
                        sl = ii + jj*5
                        ax[jj,ii].imshow(np.abs(save_img[:,:,sl]), cmap='gray',vmin=0,vmax=1)
                        ax[jj,ii].title.set_text('Slice: ' + str(sl))
                fig.tight_layout()
                plt.savefig(params["path_to_predictions"]+"ValEpoch"+str(epoch)+"_"+tag+"_batch"+str(i)+"_RECOImgafterPP_m"+str(mrsi.item()))
                plt.close(fig)


            if False:
                ### Fully Sampled Image After PP ###
                #print("### Fully Sampled Image After PP ###")
                print(img.shape)
                img_v2=img[:,0] + 1j*img[:,1]
                print(torch.amax(torch.abs(img_v2[0])))
                save_img = np.array(img_v2[0].cpu().detach())
                fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(20,25))
                fig.suptitle("Image Space, \n", fontsize=16)
                for jj in range(6):
                    for ii in range(5):
                        sl = ii + jj*5
                        ax[jj,ii].imshow(np.abs(save_img[:,:,sl]), cmap='gray',vmin=0,vmax=1)
                        ax[jj,ii].title.set_text('Slice: ' + str(sl))
                fig.tight_layout()
                plt.savefig(params["path_to_predictions"]+"ValEpoch"+str(epoch)+"_"+tag+"_batch"+str(i)+"_FSImgafterPP_m"+str(mrsi.item()))
                plt.close(fig)
            
            loss = params["loss_func"](reco_img=reco_img, 
                                        img=img, 
                                        img_under=img_under,
                                        reco_kspace=reco_kspace, 
                                        mrsiData_Cart=mrsiData_Cart, 
                                        mrsiData_Cart_under=mrsiData_Cart_under,
                                        mrsi=mrsi,
                                        mask=brainMaskAug)
            
            
            val_epoch_loss.append(loss.item())

            sto_batch = time.time() - sta_batch
            if params["verbose"]:
                log_batch = ' ~ ValEpoch'+ tag + ': {:03d}, Batch: ({:03d}/{:03d}) Loss: {:.8f}, GPU-Time: {:.4f}, Load-Time: {:.4f}'
                print(log_batch.format(epoch+1, i+1, n_batches_val_tmp, loss.item(), sto_batch, sto_time))
            
        val_epoch_loss = np.mean(np.array(val_epoch_loss))

        sto_epoch = time.time() - sta_epoch
        log_epoch = 'valEpoch'+ tag + ': {:03d}, Loss: {:.8f}, Time: {:.4f}'
        print(log_epoch.format(epoch+1, val_epoch_loss , sto_epoch))
        if writer:
            writer.add_scalars(params["model_name"] + "/loss", {"validation"+tag: val_epoch_loss}, epoch)
            writer.flush()
        
        
        del img_under, mrsiData_Cart_under, img, mrsiData_Cart, sense, sense_ext, reco_img, reco_kspace, loss
        torch.cuda.empty_cache()
        
        
    return val_epoch_loss
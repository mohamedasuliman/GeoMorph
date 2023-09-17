#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from models.GeoMorph_model import feature_extraction, geomorph
from models.Multimodal_Dataloader import MRIImages
from utils.utils import LossFuns, normalize_feats, grad_loss

""" Set your hyper-parameters """
Num_Epochs = 100
learning_rate = 2e-4
batch_size = 1

lambda_mse = 1.0
lambda_cc  = 1.0
lambda_reg = 0.5

num_labels_coar = 600 
num_feat= [32,32,64,64,128]
dec_num_feat = [2*num_feat[-1],128,128,128,num_labels_coar]

data_ico =6
labels_ico_coar =6
control_ico_coar=4

ver_dic = {0:12,1:42,2:162,3:642,4:2562,5:10242,6:40962}
target_index = 3
targets_dic = {1:'RSNs_Myelin_Topo',2:'RSNs_Topo',3:'RSNs_Myelin',4:'Myelin_Topo'} # Type of input features
target_type = targets_dic[target_index]
in_channels= 33 

loss_pen ='l1'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_best = True
saved_model_dir = 'saved_models/'

RSNs_myelin_indices = list(range(1,25))+[26,27,29,30,31,33,34,40] 
################################################################
print("The device is '{}' ".format(device))

model_feat = feature_extraction(in_ch=in_channels, 
                                num_features=num_feat, 
                                device=device, 
                                data_ico=data_ico)
model_feat.to(device)
####################################
model = geomorph(dec_num_features=dec_num_feat,
                                     num_neigh=num_labels_coar, 
                                     device=device,
                                     data_ico=data_ico, 
                                     labels_ico=labels_ico_coar,
                                     control_ico=control_ico_coar)
model.to(device)

print("The Feat model has {} paramerters".format(sum(x.numel() for x in model_feat.parameters())))
print("The Coarse model has {} paramerters".format(sum(x.numel() for x in model.parameters())))
####################################

optimizer =torch.optim.Adam(list(model_feat.parameters()) + list(model.parameters()), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=1e-7)


############################
### Set your directories ###
############################
ddr_files_dir = 'DDR_files/'  # DDR files directory
moving_dir = 'moving_images/'  # moving imgs location
target_dir = 'target_images/'  # target imgs location

root1 = 'Data_files2/UKB_HCP/Test_Red2/Subjects_ID_fold1'
root2 = 'Data_files2/UKB_HCP/Test_Red2/Subjects_ID_fold2' 
root3 = 'Data_files2/UKB_HCP/Test_Red2/Subjects_ID_fold3' 
root4 = 'Data_files2/UKB_HCP/Test_Red2/Subjects_ID_fold4'
root5_UKB = 'Data_files2/UKB_HCP/Test_Red2/Subjects_ID_fold5_UKB'
root5_HCP = 'Data_files2/UKB_HCP/Test_Red2/Subjects_ID_fold5_HCP'

# moving imgs Ids files
Id_file_t1 = ddr_files_dir+'Subjects_IDs/Subjects_ID_1'
Id_file_t2 = ddr_files_dir+'Subjects_IDs/Subjects_ID_2'
Id_file_t3 = ddr_files_dir+'Subjects_IDs/Subjects_ID_3' 
Id_file_val  = ddr_files_dir+'Subjects_IDs/Subjects_ID_val'
Id_file_test = ddr_files_dir+'Subjects_IDs/Subjects_ID_test' # if testing == True


edge_in=torch.LongTensor(np.load(ddr_files_dir+'edge_index_'+str(data_ico)+'.npy')).to(device)
hex_in =torch.LongTensor(np.load(ddr_files_dir+'hexagons_'+str(data_ico)+'.npy')).to(device)

train_dataset = MRIImages(moving_dir,target_dir,target_type,
                          Id_file1=Id_file_t1, Id_file2=Id_file_t2, Id_file3=Id_file_t3)
    
val_dataset = MRIImages(moving_dir,target_dir,target_type,
                        Id_file1=Id_file_val)


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                               shuffle=True, pin_memory=True)

val_dataloader= torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                                shuffle=True, pin_memory=True)

print('Number of Train Images = ',len(train_dataloader)) 
print('Number of Val Images  = ',len(val_dataloader))
print('\n')

num_train_data = len(train_dataloader)

def validation(dataloader, edge_in): 
    
    model_feat.eval()
    model.eval()   
    
    val_losses = torch.zeros((len(dataloader),1))
    val_loss_cc = torch.zeros((len(dataloader),1))
    val_loss_l2= torch.zeros((len(dataloader),1)) 
    
    for batch_idx, (moving_ims_t, target_ims_t) in enumerate(dataloader):       
        moving_ims_t, target_ims_t = (moving_ims_t.squeeze(0)).to(device), (target_ims_t.squeeze(0)).to(device)

        moving_ims = normalize_feats(moving_ims_t[:,RSNs_myelin_indices])
        target_ims = normalize_feats(target_temp=target_ims_t[:,RSNs_myelin_indices]) 
        
        with torch.no_grad():
            val_feat_x, val_feat_y = model_feat(moving_ims,target_ims, edge_in)
            warped_moving_val,_,_ = model(moving_ims,target_ims, val_feat_x, val_feat_y)                        
                        
            val_loss_cc_RSNs, val_loss_mse_RSNs = LossFuns(warped_moving_val[:,0:-1],
                                                              target_ims[:,0:-1])      
            
            val_loss_cc_Myelin, val_loss_mse_Myelin = LossFuns(warped_moving_val[:,-1].reshape(1,-1),
                                                                  target_ims[:,-1].reshape(1,-1))
            
            val_loss_cc[batch_idx,:]= val_loss_cc_RSNs+val_loss_cc_Myelin
            val_loss_l2[batch_idx,:]= val_loss_mse_RSNs+val_loss_mse_Myelin
            
            val_losses[batch_idx,:] =val_loss_cc[batch_idx,:]+ val_loss_l2[batch_idx,:]
    
    return val_losses, val_loss_cc, val_loss_l2


def train():
    
    train_loss_all = torch.zeros(Num_Epochs,1)
    train_loss_mse= torch.zeros(Num_Epochs,1)
    train_loss_cc= torch.zeros(Num_Epochs,1)
     
    val_loss_mean_all= torch.zeros(Num_Epochs,1)
    val_loss_mean_main= torch.zeros(Num_Epochs,1)
    val_loss_mean_cc= torch.zeros(Num_Epochs,1)
    
    best_val=1e10

    
    for epoch in range(Num_Epochs):
            
    
        running_losses= 0
        running_losses_main= 0
        running_losses_corr= 0
        
        for batch_idx, (moving_ims_t, target_ims_t) in enumerate(train_dataloader):
            
            model_feat.train()
            model.train()
    
            moving_ims_t, target_ims_t = (moving_ims_t.squeeze(0)).to(device), (target_ims_t.squeeze(0)).to(device)
            
    
            moving_ims = normalize_feats(moving_ims_t[:,RSNs_myelin_indices])
            target_ims = normalize_feats(target_ims_t[:,RSNs_myelin_indices])
            
            optimizer.zero_grad() 
            
            
            train_feat_x, train_feat_y = model_feat(moving_ims, target_ims, edge_in)
            
            warped_moving_train,_, warps_train= model(moving_ims, target_ims, train_feat_x, train_feat_y)
    
            loss_cc_RSNs,loss_mse_RSNs = LossFuns(warped_moving_train[:,0:-1],
                                                     target_ims[:,0:-1])
            
            loss_cc_Myelin,loss_mse_Myelin = LossFuns(warped_moving_train[:,-1].reshape(1,-1),
                                                         target_ims[:,-1].reshape(1,-1))
            
            loss_cc =loss_cc_RSNs+loss_cc_Myelin
            loss_mse =loss_mse_RSNs+loss_mse_Myelin
            
            loss_sm= grad_loss(warps_train,hex_in,penalty=loss_pen)
                    
            loss = (lambda_mse*loss_mse+lambda_cc*loss_cc)+lambda_reg*loss_sm
            
    
            loss.backward()
            
            optimizer.step() 
            
            running_losses+=loss.item()
            running_losses_main+=loss_mse.item()
            running_losses_corr+=loss_cc.item()
    
        train_loss_all[epoch]  = torch.tensor(running_losses/num_train_data)
        train_loss_mse[epoch] = torch.tensor(running_losses_main/num_train_data)
        train_loss_cc[epoch] = torch.tensor(running_losses_corr/num_train_data)
           
        val_losses_all,val_losses_cc,val_losses_main = validation(val_dataloader, edge_in)
        val_loss_mean_all[epoch]  = torch.mean(val_losses_all, axis=0)
        val_loss_mean_cc[epoch]  = torch.mean(val_losses_cc, axis=0)
        val_loss_mean_main[epoch] = torch.mean(val_losses_main, axis=0)
    
              
        print('(Ep: {}) = (T.L = {:.4}) === (T.M = {:.4})\n ******** (V.L = {:.5}) == (V.M = {:.5}) == (V.C = {:.5})'
              .format(epoch, train_loss_all[epoch].numpy()[0],
                      train_loss_mse[epoch].numpy()[0],  val_loss_mean_all[epoch].numpy()[0],
                      val_loss_mean_cc[epoch].numpy()[0], val_loss_mean_main[epoch].numpy()[0]))
        
        scheduler.step(val_loss_mean_all[epoch])
        
    
    ##################################
        
        if save_best:
            if val_loss_mean_all[epoch].numpy()[0] < best_val:
                best_val = val_loss_mean_all[epoch].numpy()[0]
                
                torch.save(model.state_dict(), saved_model_dir+'best_val_model.pkl')
                torch.save(model_feat.state_dict(), saved_model_dir+'best_val_feat_model.pkl')
        
               

    


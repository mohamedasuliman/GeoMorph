#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np

import torch_geometric.nn as gnn

from models.GeoMorph_stn import STN

####################################################################################
ver_dic = {0:12,1:42,2:162,3:642,4:2562,5:10242,6:40962, 7:163842, 8:655362}
dim_list =[12, 42 ,162,642,2562,10242,40962]

ddr_files_dir = 'DDR_files/'  # DDR files directory

def load_ddr_var(data_ico):
    global hexes
    global edge_indexes
    global pseudos
    global upsamples
    
    hexes=[]    
    for i in range(data_ico):
        hexes.append(torch.LongTensor(np.load(ddr_files_dir+'hexagons_'+str(data_ico-i)+'.npy')))
    
    edge_indexes=[]
    for i in range(data_ico):
        edge_indexes.append(torch.LongTensor(np.load(ddr_files_dir+'edge_index_'+str(data_ico-i)+'.npy')))
    
    pseudos=[]
    for i in range(data_ico):
        pseudos.append(torch.LongTensor(np.load(ddr_files_dir+'pseudo_'+str(data_ico-i)+'.npy')))
    
    upsamples=[]
    for i in range(data_ico):
        upsamples.append(torch.LongTensor(np.load(ddr_files_dir+'upsample_to_ico'+str(data_ico-i)+'.npy')))
        
#############################################################################################
def gmmconv(inchans, outchans, kernel_size=10):
    return gnn.GMMConv(inchans, outchans, dim=2, kernel_size=kernel_size)
    

class hex_upsample(nn.Module):
    def __init__(self, ico_level, data_ico, device):
        super(hex_upsample, self).__init__()
        
        self.hex = hexes[data_ico-ico_level].to(device)
        self.upsample = upsamples[data_ico-ico_level].to(device)
        self.device =  device       
        
    def forward(self, ico_feat):
        n_ver = int(ico_feat.shape[0])
        up_ico_feat = torch.zeros(self.hex.shape[0],ico_feat.shape[1]).to(self.device)
        up_ico_feat[:n_ver] = ico_feat
        up_ico_feat[n_ver:] = torch.mean(ico_feat[self.upsample],dim=1)
        
        return up_ico_feat
    
    
class hex_pooling_mean(nn.Module):
    def __init__(self, ico_level, device):
        super(hex_pooling_mean, self).__init__()
        self.hex = hexes[ico_level].to(device)
       
    def forward(self, x):
        num_nodes = int((x.size()[0]+6)/4)
        feat_num = x.size()[1]

        x = x[self.hex[0:num_nodes]].view(num_nodes, feat_num, 7)
        x = torch.mean(x, 2)
        
        assert(x.size() == torch.Size([num_nodes, feat_num]))
                
        return x     

class hex_pooling_max(nn.Module):
    def __init__(self, ico_level, device):
        super(hex_pooling_max, self).__init__()
        self.hex = hexes[ico_level].to(device)

        
    def forward(self, x):
        num_nodes = int((x.size()[0]+6)/4)
        feat_num = x.size()[1]

        x = x[self.hex[0:num_nodes]].view(num_nodes, feat_num, 7)
        x = torch.max(x, 2)[0]
        
        assert(x.size() == torch.Size([num_nodes, feat_num]))
        
        return x 
                


class feature_extraction(nn.Module):
    def __init__(self, in_ch, num_features, device, data_ico, conv_style=gmmconv,
                 activation_function=nn.LeakyReLU(0.2, inplace=True)):
        super(feature_extraction, self).__init__()
        
        self.conv_style  = conv_style       
        self.in_channels = in_ch
        
        self.device = device
        self.data_ico = data_ico
        
        load_ddr_var(data_ico)
        
        
        self.conv1  = conv_style(self.in_channels, num_features[0])
        self.conv1s = conv_style(num_features[0], num_features[0])
        
        self.conv2  = conv_style(self.in_channels+2*num_features[0], num_features[1])
        self.conv2s  = conv_style(num_features[1], num_features[1])
               
        self.conv3  = conv_style(self.in_channels+2*num_features[1], num_features[2])
        self.conv3s  = conv_style(num_features[2], num_features[2])
        
        self.conv1_d  = conv_style(self.in_channels, num_features[0])
        self.conv1s_d = conv_style(num_features[0], num_features[0])
        
        self.conv2_d  = conv_style(self.in_channels+2*num_features[0], num_features[1])
        self.conv2s_d  = conv_style(num_features[1], num_features[1])
           
        self.conv3_d  = conv_style(self.in_channels+2*num_features[1], num_features[2])
        self.conv3s_d  = conv_style(num_features[2], num_features[2])
  
        self.conv4  = conv_style(self.in_channels+2*num_features[2], num_features[3])
        self.conv4s  = conv_style(num_features[3], num_features[3])
        
        self.conv5  = conv_style(self.in_channels+2*num_features[3], num_features[4])
        self.conv5s  = conv_style(num_features[4], num_features[4])
        

              
        self.pool1 = hex_pooling_max(0, self.device)
        self.pool2 = hex_pooling_max(1, self.device)
        self.pool3 = hex_pooling_max(2, self.device)
        self.pool4 = hex_pooling_max(3, self.device)
        self.pool5 = hex_pooling_max(4, self.device)
        
        
        self.upsample1 = hex_upsample(data_ico-5, data_ico, device)
        self.upsample2 = hex_upsample(data_ico-4, data_ico, device)
        self.upsample3 = hex_upsample(data_ico-3, data_ico, device)
        self.upsample4 = hex_upsample(data_ico-2, data_ico, device)
        self.upsample5 = hex_upsample(data_ico-1, data_ico, device)
        self.upsample6 = hex_upsample(data_ico, data_ico, device)

        self.activation_function = activation_function
        
    def forward(self, moving, target, edge_input):
        
        ######  ico-6 #####
        x = self.conv1(moving,edge_input, pseudos[0].to(self.device))
        x = self.activation_function(x)       
        x = self.conv1s(x,edge_input, pseudos[0].to(self.device)) #ico6
        x = self.activation_function(x)
        
        x_pool= self.pool1(x)
         
        x = torch.cat([x[0:dim_list[self.data_ico-1],:], x_pool, moving[0:dim_list[self.data_ico-1],:]], dim=1)
        
        ######  ico-5 #####
        x = self.conv2(x,edge_indexes[1].to(self.device), pseudos[1].to(self.device))
        x = self.activation_function(x)        
        x = self.conv2s(x,edge_indexes[1].to(self.device), pseudos[1].to(self.device)) #ico6
        x = self.activation_function(x)
        
        x_pool= self.pool2(x)
        
        x = torch.cat([x[0:dim_list[self.data_ico-2],:], x_pool, moving[0:dim_list[self.data_ico-2],:]], dim=1)
         # 
        ######  ico-4 #####
        x = self.conv3(x,edge_indexes[2].to(self.device), pseudos[2].to(self.device))
        x = self.activation_function(x)        
        x = self.conv3s(x,edge_indexes[2].to(self.device), pseudos[2].to(self.device)) #ico6
        x = self.activation_function(x)
        
        x_pool= self.pool3(x)
        
        x = torch.cat([x[0:dim_list[self.data_ico-3],:], x_pool, moving[0:dim_list[self.data_ico-3],:]], dim=1)
               
        ######  ico-3 #####
        x = self.conv4(x,edge_indexes[3].to(self.device), pseudos[3].to(self.device))
        x = self.activation_function(x)       
        x = self.conv4s(x,edge_indexes[3].to(self.device), pseudos[3].to(self.device)) #ico6
        x = self.activation_function(x)
               
        x_pool= self.pool4(x)
        
        x = torch.cat([x[0:dim_list[self.data_ico-4],:], x_pool, moving[0:dim_list[self.data_ico-4],:]], dim=1)
               
        ######  ico-2 #####
        x = self.conv5(x,edge_indexes[4].to(self.device), pseudos[4].to(self.device))
        x = self.activation_function(x)       
        x = self.conv5s(x,edge_indexes[4].to(self.device), pseudos[4].to(self.device)) #ico6
        
        feat_x = self.activation_function(x)
        
        
        ################################################################
        # Target image feat extraction
        
        y = self.conv1_d(target,edge_input, pseudos[0].to(self.device))
        y = self.activation_function(y)       
        y = self.conv1s_d(y,edge_input, pseudos[0].to(self.device)) #ico6
        y = self.activation_function(y)
        
        y_pool= self.pool1(y)
        
        y = torch.cat([y[0:dim_list[self.data_ico-1],:], y_pool, target[0:dim_list[self.data_ico-1],:]], dim=1)
        
        ######  ico-5 #####
        y = self.conv2_d(y,edge_indexes[1].to(self.device), pseudos[1].to(self.device))
        y = self.activation_function(y)        
        y = self.conv2s_d(y,edge_indexes[1].to(self.device), pseudos[1].to(self.device)) #ico6
        y = self.activation_function(y)
        
        y_pool= self.pool2(y)
                
        y = torch.cat([y[0:dim_list[self.data_ico-2],:], y_pool, target[0:dim_list[self.data_ico-2],:]], dim=1)
        
        
        ######  ico-4 #####
        y = self.conv3_d(y,edge_indexes[2].to(self.device), pseudos[2].to(self.device))
        y = self.activation_function(y)        
        y = self.conv3s_d(y,edge_indexes[2].to(self.device), pseudos[2].to(self.device)) #ico6
        y = self.activation_function(y)
        
        y_pool= self.pool3(y)
        
        y = torch.cat([y[0:dim_list[self.data_ico-3],:], y_pool, target[0:dim_list[self.data_ico-3],:]], dim=1)
               
        ######  ico-3 #####
        y = self.conv4(y,edge_indexes[3].to(self.device), pseudos[3].to(self.device))
        y = self.activation_function(y)       
        y = self.conv4s(y,edge_indexes[3].to(self.device), pseudos[3].to(self.device)) #ico6
        y = self.activation_function(y)
        
        y_pool= self.pool4(y)
        
        y = torch.cat([y[0:dim_list[self.data_ico-4],:], y_pool, target[0:dim_list[self.data_ico-4],:]], dim=1)
               
        ######  ico-2 #####
        y = self.conv5(y,edge_indexes[4].to(self.device), pseudos[4].to(self.device))
        y = self.activation_function(y)       
        y = self.conv5s(y,edge_indexes[4].to(self.device), pseudos[4].to(self.device)) #ico6
        
        feat_y = self.activation_function(y)
                
        return feat_x, feat_y
    
        
class geomorph(nn.Module):
    def __init__(self, dec_num_features, num_neigh, device, data_ico, labels_ico, 
                 control_ico):
        super(geomorph, self).__init__()
        
        
        self.device = device
        self.num_neigh = num_neigh
   
        self.upsample3 = hex_upsample(data_ico-3, data_ico, device)
        self.upsample4 = hex_upsample(data_ico-2, data_ico, device)
        self.upsample5 = hex_upsample(data_ico-1, data_ico, device)
        self.upsample6 = hex_upsample(data_ico, data_ico, device)
        
        self.res1=  ResnetBlockFC(dec_num_features[0],dec_num_features[1], self.device, data_ico)
        self.res2=  ResnetBlockFC(dec_num_features[1],dec_num_features[2], self.device, data_ico)
        self.res3=  ResnetBlockFC(dec_num_features[2],dec_num_features[3], self.device, data_ico)
        self.res4=  ResnetBlockFC(dec_num_features[3],dec_num_features[4], self.device, data_ico)
        self.res5=  ResnetBlockFC(dec_num_features[4],dec_num_features[4], self.device, data_ico)
        load_ddr_var(data_ico)
        
        
        self.down = nn.ModuleList([]) 
        self.n_res = data_ico-control_ico
        
        for i in range(self.n_res): 
            self.down.append(hex_pooling_mean(i, self.device))
            
        
        self.transformer = STN(data_ico=data_ico,labels_ico=labels_ico, 
                               control_ico=control_ico, num_neigh=self.num_neigh,
                               device=self.device)
        
        self.fc_softmax = nn.Softmax(dim=1)


    def forward(self, moving_img, target_img, feat_x, feat_y):
                
            
        feat_x = feat_x / feat_x.norm(dim=1, keepdim=True)
        feat_y = feat_y / feat_y.norm(dim=1, keepdim=True)
        
        all_feat =  torch.cat([feat_x, feat_y], dim=1)
  
        
        feat_out= self.res1(all_feat)
        feat_out = self.upsample3(feat_out)
            
        
        feat_out= self.res2(feat_out)
        feat_out = self.upsample4(feat_out)
        
        
        feat_out= self.res3(feat_out)
        feat_out = self.upsample5(feat_out)
        
        
        feat_out= self.res4(feat_out)
        feat_out = self.upsample6(feat_out)
        
        
        feat_out= self.res5(feat_out)
        
        ##### End of Decoder ###
        
        for i in range(self.n_res):
            feat_out= self.down[i](feat_out)
            
        deff_idxs = self.fc_softmax(feat_out)
        
        
        warped_moving_img, deformed_control_ico, warps_moving  = self.transformer(moving_img,target_img,deff_idxs)
     
        return warped_moving_img, deformed_control_ico, warps_moving
        
  

class ResnetBlockFC(nn.Module):

    def __init__(self, size_in, size_out, device, data_ico, size_hidden=None, 
                 conv_style=gmmconv, actication_fun=nn.ReLU()):
        super().__init__()
        
        self.size_in = size_in
        self.size_hidden = size_hidden
        self.size_out = size_out
        self.data_ico = data_ico
        self.device = device
        
        if size_out is None:
            size_out = size_in

        if size_hidden is None:
            size_hidden = min(size_in, size_out)


        self.conv1 = conv_style(size_in, size_hidden)
        self.conv2 = conv_style(size_hidden, size_out)
        
        self.actication_fun = actication_fun

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = conv_style(size_in, size_out)
            

    def forward(self, input_feat):
        
        feat_level = self.data_ico-dim_list.index(input_feat.shape[0])
        
        
        x = self.conv1(input_feat,edge_indexes[feat_level].to(self.device),pseudos[feat_level].to(self.device))
        x = self.actication_fun(x)
        x = self.conv2(x,edge_indexes[feat_level].to(self.device),pseudos[feat_level].to(self.device))
        
        if self.shortcut is not None:
            x_s = self.shortcut(input_feat,edge_indexes[feat_level].to(self.device),pseudos[feat_level].to(self.device))
        else:
            x_s = input_feat
            
        x_out = self.actication_fun(x+x_s)
        
        return x_out



    
    

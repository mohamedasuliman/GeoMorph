#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import torch

 
class MRIImages(torch.utils.data.Dataset):

    def __init__(self, moving_dir, target_dir,
                 target_type, Id_file1=None, Id_file2=None, Id_file3=None):

        self.subjects = []
        if Id_file1 is not None:
            self.subjects = self.subjects + open(Id_file1, "r").read().splitlines()
        if Id_file2 is not None:
            self.subjects = self.subjects + open(Id_file2, "r").read().splitlines()
        if Id_file3 is not None:
            self.subjects = self.subjects + open(Id_file3, "r").read().splitlines()
            
        self.moving_images = [] 
        self.target_images = [] 
        
        self.target_img=(torch.load(target_dir+'Matrices/'+target_type+'.pt').to(torch.float32))
        
        for subject_id in self.subjects:
            moving_image =(torch.load(moving_dir+'Matrices/'+str(subject_id)+'.pt').to(torch.float32))
            self.moving_images.append(moving_image)            
               
    def __getitem__(self, index):
        
        moving_img = self.moving_images[index]
        target_img = self.target_img
    
        return moving_img, target_img

    def __len__(self):
        return len(self.subjects)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: An Tao, Pengliang Ji, Ziyi Wu
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn, dazitu616@gmail.com
@Time: 2022/7/30 7:49 PM

MODIFIED TO CONTAIN ONLY MODELNET40 CLASSIFICATION DATASET
"""

import torch
import os
import glob
import numpy as np
from copy import copy
from torch.utils.data import Dataset
import random
from stl import mesh

LABELS = ['bareteeth', 'cheeks_in', 'eyebrow', 'high_smile',
          'lips_back', 'lips_up', 'mouth_down', 'mouth_extreme',
          'mouth_middle', 'mouth_open', 'mouth_side', 'mouth_up']
LABEL_STR2INT = {label:i for i,label in enumerate(LABELS)}

TRAIN_SET = ['FaceTalk_170725_00137_TA', 'FaceTalk_170725_00137_TA_n1', 'FaceTalk_170725_00137_TA_n2', 'FaceTalk_170725_00137_TA_n3', 'FaceTalk_170725_00137_TA_n4',
             'FaceTalk_170728_03272_TA', 'FaceTalk_170728_03272_TA_n1', 'FaceTalk_170728_03272_TA_n2', 'FaceTalk_170728_03272_TA_n3', 'FaceTalk_170728_03272_TA_n4',
             'FaceTalk_170731_00024_TA', 'FaceTalk_170731_00024_TA_n1', 'FaceTalk_170731_00024_TA_n2', 'FaceTalk_170731_00024_TA_n3', 'FaceTalk_170731_00024_TA_n4',
             'FaceTalk_170809_00138_TA', 'FaceTalk_170809_00138_TA_n1', 'FaceTalk_170809_00138_TA_n2', 'FaceTalk_170809_00138_TA_p1', 'FaceTalk_170809_00138_TA_p2',
             'FaceTalk_170811_03274_TA', 'FaceTalk_170811_03274_TA_n1', 'FaceTalk_170811_03274_TA_n2', 'FaceTalk_170811_03274_TA_n3', 'FaceTalk_170811_03274_TA_n4',
             'FaceTalk_170811_03275_TA', 'FaceTalk_170811_03275_TA_n1', 'FaceTalk_170811_03275_TA_n2', 'FaceTalk_170811_03275_TA_n3', 'FaceTalk_170811_03275_TA_n4',
             'FaceTalk_170908_03277_TA', 'FaceTalk_170908_03277_TA_n1', 'FaceTalk_170908_03277_TA_n2', 'FaceTalk_170908_03277_TA_n3', 'FaceTalk_170908_03277_TA_n4',
             'FaceTalk_170912_03278_TA', 'FaceTalk_170912_03278_TA_n1', 'FaceTalk_170912_03278_TA_n2', 'FaceTalk_170912_03278_TA_n3', 'FaceTalk_170912_03278_TA_n4',
             'FaceTalk_170913_03279_TA', 'FaceTalk_170913_03279_TA_n1', 'FaceTalk_170913_03279_TA_n2', 'FaceTalk_170913_03279_TA_n3', 'FaceTalk_170913_03279_TA_n4',
             'FaceTalk_170915_00223_TA', 'FaceTalk_170915_00223_TA_n1', 'FaceTalk_170915_00223_TA_n2', 'FaceTalk_170915_00223_TA_n3', 'FaceTalk_170915_00223_TA_n4']

TEST_SET = ['FaceTalk_170904_00128_TA', 'FaceTalk_170904_00128_TA_n1', 'FaceTalk_170904_00128_TA_n2', 'FaceTalk_170904_00128_TA_n3', 'FaceTalk_170904_00128_TA_n4',
             'FaceTalk_170904_03276_TA', 'FaceTalk_170904_03276_TA_n1', 'FaceTalk_170904_03276_TA_n2', 'FaceTalk_170904_03276_TA_n3', 'FaceTalk_170904_03276_TA_n4']

def translate_pointcloud(pointcloud):
    xyz1 = (3./2. - 2./3) * torch.rand(3, dtype=pointcloud.dtype) + 2./3
    xyz2 = (0.2 - (-0.2)) * torch.rand(3, dtype=pointcloud.dtype) + (-0.2)
    
    translated_pointcloud = torch.add(pointcloud*xyz1, xyz2)
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(torch.from_numpy(rotation_matrix)) # random rotation (x,z)
    return pointcloud



def load_data_cls(partition, process_type):
    assert partition in ['train', 'test']
    assert process_type in ['eyeless', 'front']
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data', 'Coma_peaks')
    DATA_DIR = os.path.abspath("/mnt/data/yusuf/e-mesh-attack/coma_expanded_2")
    all_data = []
    selected_set = TRAIN_SET if partition=='train' else TEST_SET
    for folder in selected_set:
        for label_type in LABELS:
            file_dir = os.path.join(*[DATA_DIR, folder, label_type])
            file_path = glob.glob(file_dir+f'/*{process_type}.stl')[0]
            mesh_data = mesh.Mesh.from_file(file_path)
            all_data.append({
                'meshvectors': mesh_data.vectors,
                'meshnormals': mesh_data.normals,
                'cate': LABEL_STR2INT[label_type],
                'pc': mesh_data.vectors.mean(axis=1),
                'shift': 0,
                'scale': 1
            })
    return all_data

class Coma(Dataset):
    def __init__(self, partition='train', scale_mode="unit_sphere", process_type='eyeless'):
        self.partition, self.process_type = partition, process_type
        self.all_data = load_data_cls(partition, process_type)
        self.stats = self.get_statistics()
        assert scale_mode is None or scale_mode in ("none", 'global_unit', "unit_sphere")
        self.scale_mode = scale_mode
        self.label_dict = {v:k for k,v in LABEL_STR2INT.items()}
    
    # Get global statistics  
    def get_statistics(self):
        # Use train split to get statistics
        pc_list = [data['pc'] for data in self.all_data]
        point_list = np.concatenate(pc_list, axis=0)
        mean = point_list.mean(axis=0) # (1, 3)
        std = point_list.std()        # (1, )
        self.stats = {'mean': mean, 'std': std}
        return self.stats
    
    # For a given pc, return shift, scale that translates to origin, fits to unit ball, pc: (N,D)
    @staticmethod
    def normalize_pc(pc):
        shift = pc.mean(axis=0, keepdims=True)
        norm_pc = pc - shift
        scale = np.linalg.norm(norm_pc, axis=1, keepdims=True).max()
        return shift, scale
    
    def __getitem__(self, item):
        # Copy original
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.all_data[item].items()}
        pointcloud = data["pc"]
        # Normalization
        shift, scale = 0, 1
        if self.scale_mode=='unit_sphere':
            shift, scale = self.normalize_pc(pointcloud)
        data["shift"], data["scale"] = shift, scale
        pointcloud = (pointcloud-shift)/scale
        
        # Masking for z>0
        indices = pointcloud[:,2] > 0
        #print(indices.shape)
        #print(pointcloud.shape)
        pointcloud = pointcloud[indices,:]
        #print(pointcloud.shape)


        # pointcloud = translate_pointcloud(torch.tensor(pointcloud))

        data["meshvectors"] = data["meshvectors"][indices]
        data["meshnormals"] = data["meshnormals"][indices]
        # data["pc"] = pointcloud
        
        # return data
        
        pointcloud = torch.from_numpy(pointcloud)
        # Random Augmentation
        if self.partition == 'train':
            # print(type(pointcloud))
            pointcloud = translate_pointcloud(pointcloud)
            # pointcloud = rotate_pointcloud(pointcloud)
            pointcloud = pointcloud[torch.randperm(pointcloud.size()[0])]
        
        data["pc"] = pointcloud.clone()
        # Translate everything to torch
        # meshes = data["mesh"].vectors
        data = {k:torch.tensor(v).clone() if (isinstance(v,int) or isinstance(v, np.ndarray))
                else copy(v) for k, v in data.items()}
        return data

    def __len__(self):
        return len(self.all_data)
    

# if __name__ == "__main__":
    
    
#     dataset = Coma()
    
#     item = dataset[5]
        
#     np.save("/home/robust/e-mesh-attack/data_attacked/e-mesh-central/all_data_5.npy", item)
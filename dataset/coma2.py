#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import os
import glob
import numpy as np
from copy import copy
from torch.utils.data import Dataset
import random
from stl import mesh
from pathlib import Path
import pathlib

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

def translate_pointcloud(pointcloud, meshvectors, meshnormals):
    xyz1 = (3./2. - 2./3) * torch.rand(3, dtype=pointcloud.dtype) + 2./3
    xyz2 = (0.2 - (-0.2)) * torch.rand(3, dtype=pointcloud.dtype) + (-0.2)
    
    translated_pointcloud = torch.add(pointcloud*xyz1, xyz2)
    
    A,B,C = meshvectors[:,0,:],meshvectors[:,1,:],meshvectors[:,2,:]
    A = torch.add(A*xyz1, xyz2).unsqueeze(1)
    B = torch.add(B*xyz1, xyz2).unsqueeze(1)
    C = torch.add(C*xyz1, xyz2).unsqueeze(1)
    translated_meshvectors = torch.cat([A,B,C], dim=1)
    translated_meshnormals = meshnormals*(1/(xyz1+1e-12))
    return translated_pointcloud, translated_meshvectors, translated_meshnormals

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
    DATA_DIR = Path("./coma_expanded_2")

    all_data = []
    selected_set = TRAIN_SET if partition=='train' else TEST_SET
    for folder in selected_set:
        for label_type in LABELS:
            file_dir = DATA_DIR / folder / label_type
            # print(file_dir)
            file_path = glob.glob(file_dir.resolve().as_posix() + f'/*{process_type}.stl')[0]
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
    def __init__(self, partition='train', scale_mode="unit_sphere", process_type='eyeless', z_filter=True):
        self.partition, self.process_type, self.z_filter = partition, process_type, z_filter
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
        meshvectors = data["meshvectors"]
        meshnormals = data["meshnormals"]
        
        # Calculate shift-scale values for unit sphere normalization
        shift, scale = 0, 1
        if self.scale_mode=='unit_sphere':
            shift, scale = self.normalize_pc(pointcloud)
        data["shift"], data["scale"] = shift, scale
        # Normalization of pc by shfit/scale
        pointcloud = (pointcloud-shift)/scale
        # Normalization of triangle points by shift/scale
        meshvectors = (meshvectors-shift)/scale
        # Normalization of normals to unit vectors
        meshnormals /= np.linalg.norm(meshnormals, ord=2, axis=1, keepdims=True)
        
        
        # Masking for z>0
        if self.z_filter:
            # print("filtered by z!")
            indices = pointcloud[:,2] > 0
            pointcloud = pointcloud[indices,:]
            meshvectors = meshvectors[indices]
            meshnormals = meshnormals[indices]
        
        # To torch
        pointcloud = torch.from_numpy(pointcloud)
        meshvectors = torch.from_numpy(meshvectors)
        meshnormals = torch.from_numpy(meshnormals)
        
        # Random Augmentation
        if self.partition == 'train':
            # print(type(pointcloud))
            pointcloud,meshvectors,meshnormals = translate_pointcloud(pointcloud,meshvectors,meshnormals)
            # pointcloud = rotate_pointcloud(pointcloud)
            indices = torch.randperm(pointcloud.size()[0])
            pointcloud = pointcloud[indices]
            meshvectors = meshvectors[indices]
            meshnormals = meshnormals[indices]
        
        data["pc"] = pointcloud.clone()
        data["meshvectors"] = meshvectors.clone()
        data["meshnormals"] = meshnormals.clone()
        # Translate everything to torch
        # meshes = data["mesh"].vectors
        data = {k:torch.tensor(v).clone() if (isinstance(v,int) or isinstance(v, np.ndarray))
                else copy(v) for k, v in data.items()}
        return data

    def __len__(self):
        return len(self.all_data)
    

if __name__ == "__main__":
    
    
    dataset = Coma()
    
    item = dataset[5]
    print(item.shape)
        
    # np.save("/home/robust/e-mesh-attack/data_attacked/e-mesh-central/all_data_5.npy", item)

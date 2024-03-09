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


LABELS = ['ANGER', 'DISGUST', 'FEAR', 
          'HAPPY', 'SADNESS', 'SURPRISE']
LABEL_STR2INT = {label:i for i,label in enumerate(LABELS)}

TRAIN_SET = ['bs006', 'bs007', 'bs008', 'bs009', 'bs010', 'bs011', 'bs012',
              'bs013', 'bs014', 'bs015', 'bs016', 'bs017', 'bs018', 'bs019',
              'bs020', 'bs021', 'bs022', 'bs023', 'bs024', 'bs025', 'bs026', 
              'bs027', 'bs028', 'bs029', 'bs030', 'bs031', 'bs032', 'bs033', 
              'bs034', 'bs035', 'bs036', 'bs037', 'bs038', 'bs039', 'bs040', 
              'bs041', 'bs042', 'bs043', 'bs044', 'bs045', 'bs046', 'bs047', 
              'bs048', 'bs049', 'bs050', 'bs051', 'bs052', 'bs053', 'bs054', 
              'bs055', 'bs056', 'bs057', 'bs058', 'bs059', 'bs060', 'bs061', 
              'bs062', 'bs063', 'bs064', 'bs065', 'bs066', 'bs067', 'bs068', 
              'bs069', 'bs070', 'bs071', 'bs072', 'bs073', 'bs074', 'bs075', 
              'bs076', 'bs077', 'bs078', 'bs079', 'bs080', 'bs081', 'bs082', 
              'bs083', 'bs084', 'bs085', 'bs086', 'bs087', 'bs088', 'bs089', 
              'bs090', 'bs091', 'bs092', 'bs093', 'bs094', 'bs095', 'bs096']

TEST_SET = ['bs000', 'bs001', 'bs002', 'bs003', 'bs004', 'bs005','bs097',
            'bs098', 'bs099', 'bs100', 'bs101', 'bs102', 'bs103', 'bs104']

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
    assert process_type in [None]
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data', 'Coma_peaks')
    DATA_DIR = Path("./Bosphorus12K")

    all_data = []
    selected_set = TRAIN_SET if partition=='train' else TEST_SET
    for i, single_path in enumerate(sorted(DATA_DIR.glob('**/*_E_*.stl'))):
        # print(i, single_path)
        person_name = single_path.stem.split("_E_")[0]
        # Check if the person is the selected TRAIN/TEST Set
        if person_name not in selected_set: continue
        # Get data and label
        label_type = single_path.stem.split("_E_")[-1][:-2]
        mesh_data = mesh.Mesh.from_file(single_path)
        all_data.append({
            'meshvectors': mesh_data.vectors,
            'meshnormals': mesh_data.normals,
            'cate': LABEL_STR2INT[label_type],
            'pc': mesh_data.vectors.mean(axis=1),
            'shift': 0,
            'scale': 1,
            'person': person_name
            })
    return all_data

class Bosphorus(Dataset):
    def __init__(self, partition='train', scale_mode="unit_sphere", process_type=None, z_filter=False):
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
    
    
    dataset = Bosphorus()
    
    item = dataset[5]
    print(item.shape)
        
    # np.save("/home/robust/e-mesh-attack/data_attacked/e-mesh-central/all_data_5.npy", item)

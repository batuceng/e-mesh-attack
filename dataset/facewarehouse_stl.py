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


LABELS = ['pose_0', 'pose_1', 'pose_2', 'pose_3', 'pose_4', 'pose_5', 'pose_6',
          'pose_7', 'pose_8', 'pose_9', 'pose_10', 'pose_11', 'pose_12', 'pose_13', 
          'pose_14', 'pose_15', 'pose_16', 'pose_17', 'pose_18', 'pose_19']
LABEL_STR2INT = {label:i for i,label in enumerate(LABELS)}

TRAIN_SET = ['Tester_001', 'Tester_002', 'Tester_003', 'Tester_004', 'Tester_005', 'Tester_006',
             'Tester_007', 'Tester_008', 'Tester_009', 'Tester_010', 'Tester_011', 'Tester_012',
             'Tester_013', 'Tester_014', 'Tester_015', 'Tester_016', 'Tester_017', 'Tester_018',
             'Tester_019', 'Tester_020', 'Tester_021', 'Tester_022', 'Tester_023', 'Tester_024',
             'Tester_031', 'Tester_032', 'Tester_033', 'Tester_034', 'Tester_035', 'Tester_036',
             'Tester_037', 'Tester_038', 'Tester_039', 'Tester_040', 'Tester_041', 'Tester_042',
             'Tester_043', 'Tester_044', 'Tester_045', 'Tester_046', 'Tester_047', 'Tester_048',
             'Tester_049', 'Tester_050', 'Tester_051', 'Tester_052', 'Tester_053', 'Tester_054',
             'Tester_055', 'Tester_056', 'Tester_057', 'Tester_058', 'Tester_059', 'Tester_060',
             'Tester_067', 'Tester_068', 'Tester_069', 'Tester_070', 'Tester_071', 'Tester_072',
             'Tester_073', 'Tester_074', 'Tester_075', 'Tester_076', 'Tester_077', 'Tester_078',
             'Tester_079', 'Tester_080', 'Tester_081', 'Tester_082', 'Tester_083', 'Tester_084',
             'Tester_085', 'Tester_086', 'Tester_087', 'Tester_088', 'Tester_089', 'Tester_090',
             'Tester_091', 'Tester_092', 'Tester_093', 'Tester_094', 'Tester_095', 'Tester_096',
             'Tester_097', 'Tester_098', 'Tester_099', 'Tester_100', 'Tester_101', 'Tester_102',
             'Tester_103', 'Tester_104', 'Tester_105', 'Tester_106', 'Tester_107', 'Tester_108',
             'Tester_115', 'Tester_116', 'Tester_117', 'Tester_118', 'Tester_119', 'Tester_120',
             'Tester_121', 'Tester_122', 'Tester_123', 'Tester_124', 'Tester_125', 'Tester_126',
             'Tester_127', 'Tester_128', 'Tester_129', 'Tester_130', 'Tester_131', 'Tester_132',
             'Tester_139', 'Tester_140', 'Tester_141', 'Tester_142', 'Tester_143', 'Tester_144',
             'Tester_145', 'Tester_146', 'Tester_147', 'Tester_148', 'Tester_149', 'Tester_150']

TEST_SET = [ 'Tester_025', 'Tester_026', 'Tester_027', 'Tester_028', 'Tester_029', 'Tester_030',
             'Tester_061', 'Tester_062', 'Tester_063', 'Tester_064', 'Tester_065', 'Tester_066',
             'Tester_109', 'Tester_110', 'Tester_111', 'Tester_112', 'Tester_113', 'Tester_114',
             'Tester_133', 'Tester_134', 'Tester_135', 'Tester_136', 'Tester_137', 'Tester_138']

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

    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data', 'Coma_peaks')
    DATA_DIR = Path("./FaceWarehouse_stl")

    all_data = []
    selected_set = TRAIN_SET if partition=='train' else TEST_SET
    for i, single_path in enumerate(sorted(DATA_DIR.glob('*/*/*.stl'))):
        #print(i, single_path)
        person_name = single_path.parent.parent.stem
        # Check if the person is the selected TRAIN/TEST Set
        if person_name not in selected_set: continue
        label_type = '_'.join(single_path.stem.split('_')[0:2])
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

class Facewarehouse(Dataset):
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
    
    all_shapes = []
    
    dataset = Facewarehouse(partition='test', z_filter=True)
    
    for i in range(len(dataset)):
        item = dataset[i]["pc"]
        all_shapes.append(item.shape[0])
    
    print(np.unique(np.array(all_shapes)))
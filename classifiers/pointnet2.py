import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from os.path import dirname, abspath

def download_pointnet2(root):
    weight_paths = {}
    DATA_DIR = os.path.join(root, 'pretrained')
    SUB_DIR = os.path.join(DATA_DIR, 'pointnet2_weights')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(SUB_DIR):
        os.mkdir(SUB_DIR)
    if not os.path.exists(os.path.join(SUB_DIR, 'best_model.pth')):
        www = 'https://github.com/yanx27/Pointnet_Pointnet2_pytorch/raw/master/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
        os.system('wget --no-check-certificate %s -P %s' % (www, SUB_DIR))
    weight_paths["best_model.pth"] = os.path.join(SUB_DIR, 'best_model.pth')
    return weight_paths

class PointNet2_cls(nn.Module):
    def __init__(self,output_channels=12,normal_channel=False):
        super(PointNet2_cls, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.InstanceNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.InstanceNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_channels)
        
        # Load
        # self.load_pretrained()
        # self.eval()

    def forward(self, xyz):
        B, _, _ = xyz.shape
        # print(f"L0 {xyz.shape}")
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm) # l1_xyz: (B, 3, N) l1_points:(B, D, N), D=128, N=512 
        # print(f"L1 xyz:{l1_xyz.shape}, points:{l1_points.shape}")
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # l1_xyz: (B, 3, N) l1_points:(B, D, N), D=256, N=128 
        # print(f"L2 xyz:{l2_xyz.shape}, points:{l2_points.shape}")
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # l1_xyz: (B, 3, N) l1_points:(B, D, N), D=256, N=128 
        # print(f"L3 xyz:{l3_xyz.shape}, points:{l3_points.shape}")
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x
    
    def forward_denoised(self, xyz, denoiser=None):
        B, _, _ = xyz.shape
        layer_data = []                         # Store each layer data
        xyz = denoiser(data=xyz, layer=0)               # Denoise Layer 0
        layer_data.append(xyz.clone().detach().requires_grad_(True))    # Store Layer 0
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        # Layer1
        l1_xyz, l1_points = self.sa1(xyz, norm) # l1_xyz: (B, 3, N) l1_points:(B, D, N), D=128, N=512 
        l1_points = denoiser(data=l1_points, layer=1)             # Denoise Layer 1
        layer_data.append(l1_points.clone().detach().requires_grad_(True))    # Store Layer 1
        # Layer2
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # l1_xyz: (B, 3, N) l1_points:(B, D, N), D=256, N=128 
        l2_points = denoiser(data=l2_points, layer=2)             # Denoise Layer 2
        layer_data.append(l2_points.clone().detach().requires_grad_(True))    # Store Layer 2
        # Layer3
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # l1_xyz: (B, 3, N) l1_points:(B, D, N), D=256, N=128 
        l3_points = denoiser(data=l3_points, layer=3)             # Denoise Layer 3
        layer_data.append(l3_points.clone().detach().requires_grad_(True))    # Store Layer 3
        # No Layer4
        layer_data.append(None)
        # Classification Head
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, layer_data
    
    def load_pretrained(self, model_name="best_model.pth", root=dirname(abspath(__file__)) ):
        assert model_name in ["best_model.pth"]
        weight_paths = download_pointnet2(root)
        weights = torch.load(weight_paths[model_name])
        self.load_state_dict(weights['model_state_dict'])
    

# PointNet++ utils
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    # Added Generator to keep algorithm Deterministic
    g_cpu = torch.Generator(device=device)
    g_cpu.manual_seed(0)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, generator=g_cpu, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.InstanceNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

if __name__ == "__main__":
    model = PointNet2_cls()
    model.load_pretrained()
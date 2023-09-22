"""
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@File: curvenet_cls.py
@Time: 2021/01/21 3:10 PM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from os.path import dirname, abspath
from time import time

curve_config = {
        'default': [[100, 5], [100, 5], None, None],
        'long':  [[10, 30], None,  None,  None]
    }

class CurveNet(nn.Module):
    def __init__(self, num_classes=40, k=20, setting='default'):
        super(CurveNet, self).__init__()

        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(npoint=1024, radius=0.05, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][0])
        
        self.cic21 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=1024, radius=0.1, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=256, radius=0.1, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=256, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=64, radius=0.2, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=64, radius=0.4, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][3])

        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.InstanceNorm1d(1024),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Linear(1024 * 2, 512, bias=False)
        self.conv2 = nn.Linear(512, num_classes) 
        self.bn1 = nn.InstanceNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        
        # Load 
        self.load_pretrained()
        self.eval()


    def forward(self, xyz):
        l0_points = self.lpfa(xyz, xyz) #l0_points: (B,D,N) [16, 32, 1024]

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points) #l1_xyz: (B,3,N) [16, 3, 1024] l1_points: (B,D,N) [16, 64, 1024]

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points) #l2_xyz: (B,3,N) [16, 3, 1024] l2_points: (B,D,N) [16, 128, 1024]

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points) #l3_xyz: (B,3,N) [16, 3, 256] l3_points: (B,D,N) [16, 256, 256]

        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points) #l4_xyz: (B,3,N) [16, 3, 64] l4_points: (B,D,N) [16, 512, 64]

        x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)
        
        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        x = F.relu(self.bn1(self.conv1(x).unsqueeze(-1)), inplace=True).squeeze(-1)
        x = self.dp1(x)
        x = self.conv2(x)
        return x

    def forward_denoised(self, xyz, denoiser=None):
        B, _, _ = xyz.shape
        layer_data = []                                                 # Store each layer data
        xyz = denoiser(data=xyz, layer=0)                               # Denoise Layer 0
        layer_data.append(xyz.clone().detach().requires_grad_(True))    # Store Layer 0

        l0_points = self.lpfa(xyz, xyz) #l0_points: (B,D,N) [16, 32, 1024]        

        # Layer 1
        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points) #l1_xyz: (B,3,N) [16, 3, 1024] l1_points: (B,D,N) [16, 64, 1024]
        l1_points = denoiser(data=l1_points, layer=1)
        layer_data.append(l1_points.clone().detach().requires_grad_(True))

        # Layer 2
        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points) #l2_xyz: (B,3,N) [16, 3, 1024] l2_points: (B,D,N) [16, 128, 1024]
        l2_points = denoiser(data=l2_points, layer=2)
        layer_data.append(l2_points.clone().detach().requires_grad_(True))

        # Layer 3
        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points) #l3_xyz: (B,3,N) [16, 3, 256] l3_points: (B,D,N) [16, 256, 256]
        l3_points = denoiser(data=l3_points, layer=3)
        layer_data.append(l3_points.clone().detach().requires_grad_(True))

        # Layer 4
        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points) #l4_xyz: (B,3,N) [16, 3, 64] l4_points: (B,D,N) [16, 512, 64]
        l4_points = denoiser(data=l4_points, layer=4)
        layer_data.append(l4_points.clone().detach().requires_grad_(True))


        x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)
        
        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        x = F.relu(self.bn1(self.conv1(x).unsqueeze(-1)), inplace=True).squeeze(-1)
        x = self.dp1(x)
        x = self.conv2(x)
        return x, layer_data
    
    def load_pretrained(self, model_name="model.t7", root=dirname(abspath(__file__)) ):
        assert model_name in ["model.t7"]
        # assert model_name in ["model.cls.1024.t7", "model.cls.2048.t7"]
        # weight_paths = download_dgcnn(root)
        weight_paths = download_curvenet(root)
        weights = torch.load(weight_paths[model_name])
        weights = {k[7:]:v for (k,v) in weights.items()} # Remove 'module' prefix from all keys. Added due to nn.DataParalel on pretrainig
        self.load_state_dict(weights)
        

def download_curvenet(root):
    weight_paths = {}
    DATA_DIR = os.path.join(root, 'pretrained')
    SUB_DIR = os.path.join(DATA_DIR, 'curvenet_weights')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(SUB_DIR):
        os.mkdir(SUB_DIR)
    if not os.path.exists(os.path.join(SUB_DIR, 'model.t7')):
        www = '1_p7Jgm-VsRmnx6Ea4qGIgayv3g2IYIm4'
        os.system('gdown --no-check-certificate %s -O %s' % (www, os.path.join(SUB_DIR, 'model.t7')))
    weight_paths["model.t7"] = os.path.join(SUB_DIR, 'model.t7')
    return weight_paths
        


# CurveNet utils

def knn(x, k):
    k = k + 1
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def normal_knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

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
    farthest = torch.randint(0, N, (B,), dtype=torch.long, generator=g_cpu, device=device) * 0
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
    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    torch.cuda.empty_cache()

    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()

    new_points = index_points(points, idx)
    torch.cuda.empty_cache()

    if returnfps:
        return new_xyz, new_points, idx
    else:
        return new_xyz, new_points

class Attention_block(nn.Module):
    '''
    Used in attention U-Net.
    '''
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm1d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.leaky_relu(g1+x1, negative_slope=0.2)
        psi = self.psi(psi)

        return psi, 1. - psi


class LPFA(nn.Module):
    def __init__(self, in_channel, out_channel, k, mlp_num=2, initial=False):
        super(LPFA, self).__init__()
        self.k = k
        self.device = torch.device('cuda')
        self.initial = initial

        if not initial:
            self.xyz2feature = nn.Sequential(
                        nn.Conv2d(9, in_channel, kernel_size=1, bias=False),
                        nn.InstanceNorm2d(in_channel))

        self.mlp = []
        for _ in range(mlp_num):
            self.mlp.append(nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, bias=False),
                                 nn.InstanceNorm2d(out_channel),
                                 nn.LeakyReLU(0.2)))
            in_channel = out_channel
        self.mlp = nn.Sequential(*self.mlp)        

    def forward(self, x, xyz, idx=None):
        x = self.group_feature(x, xyz, idx)
        x = self.mlp(x)

        if self.initial:
            x = x.max(dim=-1, keepdim=False)[0]
        else:
            x = x.mean(dim=-1, keepdim=False)

        return x

    def group_feature(self, x, xyz, idx):
        batch_size, num_dims, num_points = x.size()

        if idx is None:
            idx = knn(xyz, k=self.k)[:,:,:self.k]  # (batch_size, num_points, k)

        idx_base = torch.arange(0, batch_size, device=self.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        xyz = xyz.transpose(2, 1).contiguous() # bs, n, 3
        point_feature = xyz.view(batch_size * num_points, -1)[idx, :]
        point_feature = point_feature.view(batch_size, num_points, self.k, -1)  # bs, n, k, 3
        points = xyz.view(batch_size, num_points, 1, 3).expand(-1, -1, self.k, -1)  # bs, n, k, 3

        point_feature = torch.cat((points, point_feature, point_feature - points),
                                dim=3).permute(0, 3, 1, 2).contiguous()

        if self.initial:
            return point_feature

        x = x.transpose(2, 1).contiguous() # bs, n, c
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)  #bs, n, k, c
        x = x.view(batch_size, num_points, 1, num_dims)
        feature = feature - x

        feature = feature.permute(0, 3, 1, 2).contiguous()
        point_feature = self.xyz2feature(point_feature)  #bs, c, n, k
        feature = F.leaky_relu(feature + point_feature, 0.2)
        return feature #bs, c, n, k


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, att=None):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        self.att = None
        if att is not None:
            self.att = Attention_block(F_g=att[0],F_l=att[1],F_int=att[2])
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.InstanceNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S], skipped xyz
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S], skipped features
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        # skip attention
        if self.att is not None:
           psix, psig = self.att(interpolated_points.permute(0, 2, 1), points1)
           points1 = points1 * psix
           
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.leaky_relu(bn(conv(new_points)), 0.2)

        return new_points


class CIC(nn.Module):
    def __init__(self, npoint, radius, k, in_channels, output_channels, bottleneck_ratio=2, mlp_num=2, curve_config=None):
        super(CIC, self).__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.bottleneck_ratio = bottleneck_ratio
        self.radius = radius
        self.k = k
        self.npoint = npoint

        planes = in_channels // bottleneck_ratio

        self.use_curve = curve_config is not None
        if self.use_curve:
            self.curveaggregation = CurveAggregation(planes)
            self.curvegrouping = CurveGrouping(planes, k, curve_config[0], curve_config[1])

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels,
                      planes,
                      kernel_size=1,
                      bias=False),
            nn.InstanceNorm1d(in_channels // bottleneck_ratio),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv1d(planes, output_channels, kernel_size=1, bias=False),
            nn.InstanceNorm1d(output_channels))

        if in_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels,
                          output_channels,
                          kernel_size=1,
                          bias=False),
                nn.InstanceNorm1d(output_channels))

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.maxpool = MaskedMaxPool(npoint, radius, k)

        self.lpfa = LPFA(planes, planes, k, mlp_num=mlp_num, initial=False)

    def forward(self, xyz, x):
 
        # max pool
        if xyz.size(-1) != self.npoint:
            xyz, x = self.maxpool(
                xyz.transpose(1, 2).contiguous(), x)
            xyz = xyz.transpose(1, 2)

        shortcut = x
        x = self.conv1(x)  # bs, c', n

        idx = knn(xyz, self.k)

        if self.use_curve:
            # curve grouping
            curves = self.curvegrouping(x, xyz, idx[:,:,1:]) # avoid self-loop

            # curve aggregation
            x = self.curveaggregation(x, curves)

        x = self.lpfa(x, xyz, idx=idx[:,:,:self.k]) #bs, c', n, k

        x = self.conv2(x)  # bs, c, n

        if self.in_channels != self.output_channels:
            shortcut = self.shortcut(shortcut)

        x = self.relu(x + shortcut)

        return xyz, x


class CurveAggregation(nn.Module):
    def __init__(self, in_channel):
        super(CurveAggregation, self).__init__()
        self.in_channel = in_channel
        mid_feature = in_channel // 2
        self.conva = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convb = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convc = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convn = nn.Conv1d(mid_feature,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convl = nn.Conv1d(mid_feature,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convd = nn.Sequential(
            nn.Conv1d(mid_feature * 2,
                      in_channel,
                      kernel_size=1,
                      bias=False),
            nn.InstanceNorm1d(in_channel))
        self.line_conv_att = nn.Conv2d(in_channel,
                                       1,
                                       kernel_size=1,
                                       bias=False)

    def forward(self, x, curves):
        curves_att = self.line_conv_att(curves)  # bs, 1, c_n, c_l

        curver_inter = torch.sum(curves * F.softmax(curves_att, dim=-1), dim=-1)  #bs, c, c_n
        curves_intra = torch.sum(curves * F.softmax(curves_att, dim=-2), dim=-2)  #bs, c, c_l

        curver_inter = self.conva(curver_inter) # bs, mid, n
        curves_intra = self.convb(curves_intra) # bs, mid ,n

        x_logits = self.convc(x).transpose(1, 2).contiguous()
        x_inter = F.softmax(torch.bmm(x_logits, curver_inter), dim=-1) # bs, n, c_n
        x_intra = F.softmax(torch.bmm(x_logits, curves_intra), dim=-1) # bs, l, c_l
        

        curver_inter = self.convn(curver_inter).transpose(1, 2).contiguous()
        curves_intra = self.convl(curves_intra).transpose(1, 2).contiguous()

        x_inter = torch.bmm(x_inter, curver_inter)
        x_intra = torch.bmm(x_intra, curves_intra)

        curve_features = torch.cat((x_inter, x_intra),dim=-1).transpose(1, 2).contiguous()
        x = x + self.convd(curve_features)

        return F.leaky_relu(x, negative_slope=0.2)


class CurveGrouping(nn.Module):
    def __init__(self, in_channel, k, curve_num, curve_length):
        super(CurveGrouping, self).__init__()
        self.curve_num = curve_num
        self.curve_length = curve_length
        self.in_channel = in_channel
        self.k = k

        self.att = nn.Conv1d(in_channel, 1, kernel_size=1, bias=False)

        self.walk = Walk(in_channel, k, curve_num, curve_length)

    def forward(self, x, xyz, idx):
        # starting point selection in self attention style
        x_att = torch.sigmoid(self.att(x))
        x = x * x_att

        _, start_index = torch.topk(x_att,
                                    self.curve_num,
                                    dim=2,
                                    sorted=False)
        start_index = start_index.squeeze().unsqueeze(2)

        curves = self.walk(xyz, x, idx, start_index)  #bs, c, c_n, c_l
        
        return curves


class MaskedMaxPool(nn.Module):
    def __init__(self, npoint, radius, k):
        super(MaskedMaxPool, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.k = k

    def forward(self, xyz, features):
        sub_xyz, neighborhood_features = sample_and_group(self.npoint, self.radius, self.k, xyz, features.transpose(1,2))

        neighborhood_features = neighborhood_features.permute(0, 3, 1, 2).contiguous()
        sub_features = F.max_pool2d(
            neighborhood_features, kernel_size=[1, neighborhood_features.shape[3]]
        )  # bs, c, n, 1
        sub_features = torch.squeeze(sub_features, -1)  # bs, c, n
        return sub_xyz, sub_features


# Walk utils

def batched_index_select(input, dim, index):
	views = [input.shape[0]] + \
		[1 if i != dim else -1 for i in range(1, len(input.shape))]
	expanse = list(input.shape)
	expanse[0] = -1
	expanse[dim] = -1
	index = index.view(views).expand(expanse)
	return torch.gather(input, dim, index)

def gumbel_softmax(logits, dim, temperature=1):
    """
    ST-gumple-softmax w/o random gumbel samplings
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = F.softmax(logits / temperature, dim=dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)

    y_hard = (y_hard - y).detach() + y
    return y_hard

class Walk(nn.Module):
    '''
    Walk in the cloud
    '''
    def __init__(self, in_channel, k, curve_num, curve_length):
        super(Walk, self).__init__()
        self.curve_num = curve_num
        self.curve_length = curve_length
        self.k = k

        self.agent_mlp = nn.Sequential(
            nn.Conv2d(in_channel * 2,
                        1,
                        kernel_size=1,
                        bias=False), nn.InstanceNorm2d(1))
        self.momentum_mlp = nn.Sequential(
            nn.Conv1d(in_channel * 2,
                        2,
                        kernel_size=1,
                        bias=False), nn.InstanceNorm1d(2))

    def crossover_suppression(self, cur, neighbor, bn, n, k):
        # cur: bs*n, 3
        # neighbor: bs*n, 3, k
        neighbor = neighbor.detach()
        cur = cur.unsqueeze(-1).detach()
        dot = torch.bmm(cur.transpose(1,2), neighbor) # bs*n, 1, k
        norm1 = torch.norm(cur, dim=1, keepdim=True)
        norm2 = torch.norm(neighbor, dim=1, keepdim=True)
        divider = torch.clamp(norm1 * norm2, min=1e-8)
        ans = torch.div(dot, divider).squeeze() # bs*n, k

        # normalize to [0, 1]
        ans = 1. + ans
        ans = torch.clamp(ans, 0., 1.0)

        return ans.detach()

    def forward(self, xyz, x, adj, cur):
        bn, c, tot_points = x.size()

        # raw point coordinates
        xyz = xyz.transpose(1,2).contiguous # bs, n, 3

        # point features
        x = x.transpose(1,2).contiguous() # bs, n, c

        flatten_x = x.view(bn * tot_points, -1)
        batch_offset = torch.arange(0, bn, device=torch.device('cuda')).detach() * tot_points

        # indices of neighbors for the starting points
        tmp_adj = (adj + batch_offset.view(-1,1,1)).view(adj.size(0)*adj.size(1),-1) #bs, n, k
    
        # batch flattened indices for teh starting points
        flatten_cur = (cur + batch_offset.view(-1,1,1)).view(-1)

        curves = []

        # one step at a time
        for step in range(self.curve_length):

            if step == 0:
                # get starting point features using flattend indices
                starting_points =  flatten_x[flatten_cur, :].contiguous()
                pre_feature = starting_points.view(bn, self.curve_num, -1, 1).transpose(1,2) # bs * n, c
            else:
                # dynamic momentum
                cat_feature = torch.cat((cur_feature.squeeze(), pre_feature.squeeze()),dim=1)
                att_feature = F.softmax(self.momentum_mlp(cat_feature),dim=1).view(bn, 1, self.curve_num, 2) # bs, 1, n, 2
                cat_feature = torch.cat((cur_feature, pre_feature),dim=-1) # bs, c, n, 2
                
                # update curve descriptor
                pre_feature = torch.sum(cat_feature * att_feature, dim=-1, keepdim=True) # bs, c, n
                pre_feature_cos =  pre_feature.transpose(1,2).contiguous().view(bn * self.curve_num, -1)

            pick_idx = tmp_adj[flatten_cur] # bs*n, k
            
            # get the neighbors of current points
            pick_values = flatten_x[pick_idx.view(-1),:]

            # reshape to fit crossover suppresion below
            pick_values_cos = pick_values.view(bn * self.curve_num, self.k, c)
            pick_values = pick_values_cos.view(bn, self.curve_num, self.k, c)
            pick_values_cos = pick_values_cos.transpose(1,2).contiguous()
            
            pick_values = pick_values.permute(0,3,1,2) # bs, c, n, k

            pre_feature_expand = pre_feature.expand_as(pick_values)
            
            # concat current point features with curve descriptors
            pre_feature_expand = torch.cat((pick_values, pre_feature_expand),dim=1)
            
            # which node to pick next?
            pre_feature_expand = self.agent_mlp(pre_feature_expand) # bs, 1, n, k

            if step !=0:
                # cross over supression
                d = self.crossover_suppression(cur_feature_cos - pre_feature_cos,
                                               pick_values_cos - cur_feature_cos.unsqueeze(-1), 
                                               bn, self.curve_num, self.k)
                d = d.view(bn, self.curve_num, self.k).unsqueeze(1) # bs, 1, n, k
                pre_feature_expand = torch.mul(pre_feature_expand, d)

            pre_feature_expand = gumbel_softmax(pre_feature_expand, -1) #bs, 1, n, k

            cur_feature = torch.sum(pick_values * pre_feature_expand, dim=-1, keepdim=True) # bs, c, n, 1

            cur_feature_cos = cur_feature.transpose(1,2).contiguous().view(bn * self.curve_num, c)

            cur = torch.argmax(pre_feature_expand, dim=-1).view(-1, 1) # bs * n, 1

            flatten_cur = batched_index_select(pick_idx, 1, cur).squeeze() # bs * n

            # collect curve progress
            curves.append(cur_feature)

        return torch.cat(curves,dim=-1)



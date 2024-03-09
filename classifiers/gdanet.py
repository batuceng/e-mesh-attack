import torch
from torch import nn
import torch.nn.functional as F
import os
from os.path import dirname, abspath

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
        
class GDANet_cls(nn.Module):
    def __init__(self, 
                 args=AttrDict({}), 
                 output_channels=20, pretrained=False):
        super(GDANet_cls, self).__init__()
        
        self.args = args
        
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn11 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn12 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn21 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn22 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn3 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn31 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn32 = nn.BatchNorm1d(128, momentum=0.1)

        self.bn4 = nn.BatchNorm1d(512, momentum=0.1)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),
                                   self.bn1)
        self.conv11 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn11)
        self.conv12 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=True),
                                    self.bn12)

        self.conv2 = nn.Sequential(nn.Conv2d(67 * 2, 64, kernel_size=1, bias=True),
                                   self.bn2)
        self.conv21 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn21)
        self.conv22 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=True),
                                    self.bn22)

        self.conv3 = nn.Sequential(nn.Conv2d(131 * 2, 128, kernel_size=1, bias=True),
                                   self.bn3)
        self.conv31 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True),
                                    self.bn31)
        self.conv32 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=True),
                                    self.bn32)

        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, bias=True),
                                   self.bn4)

        self.SGCAM_1s = SGCAM(64)
        self.SGCAM_1g = SGCAM(64)
        self.SGCAM_2s = SGCAM(64)
        self.SGCAM_2g = SGCAM(64)

        # Removed BN since Batch size is 1
        self.linear1 = nn.Linear(1024, 512, bias=True)
        # self.bn6 = nn.BatchNorm1d(512)
        self.bn6 = nn.Identity(512)
        self.dp1 = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(512, 256, bias=True)
        # self.bn7 = nn.BatchNorm1d(256)
        self.bn7 = nn.Identity(256)
        self.dp2 = nn.Dropout(p=0.4)
        self.linear3 = nn.Linear(256, output_channels, bias=True)
        
        self.pretrained = pretrained
        # Load 
        if self.pretrained is not False:
            if self.pretrained is True:
                dataset_name = 'coma_trained'
            else:
                dataset_name = self.pretrained
            self.load_pretrained(dataset_name=dataset_name)
            print("Loaded pretrained model!")
            self.eval()
        
    def load_pretrained(self, dataset_name, model_name="best_gdanet.pt", root=dirname(abspath(__file__)) ):
        assert dataset_name in ["coma_trained", "bosphorus_trained", "facewarehouse_trained"]
        assert model_name in ["best_gdanet.pt"]
        # assert model_name in ["model.cls.1024.t7", "model.cls.2048.t7"]
        weight_paths = os.path.join(*[root, "pretrained", dataset_name, model_name])
        weights = torch.load(weight_paths)
        # weights = {v for (k,v) in weights.items()} # Remove 'module' prefix from all keys. Added due to nn.DataParalel on pretrainig
        
        # Partial load
        model_dict = self.state_dict()
        weights = {k:v for k, v in weights.items() if k in model_dict}
        model_dict.update(weights)
        self.load_state_dict(model_dict)

    def forward(self, x):
        B, C, N = x.size()
        ###############
        """block 1"""
        # Local operator:
        x1 = local_operator(x, k=30)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv11(x1))
        x1 = x1.max(dim=-1, keepdim=False)[0]

        # Geometry-Disentangle Module:
        x1s, x1g = GDM(x1, M=256)

        # Sharp-Gentle Complementary Attention Module:
        y1s = self.SGCAM_1s(x1, x1s.transpose(2, 1))
        y1g = self.SGCAM_1g(x1, x1g.transpose(2, 1))
        z1 = torch.cat([y1s, y1g], 1)
        z1 = F.relu(self.conv12(z1))
        ###############
        """block 2"""
        x1t = torch.cat((x, z1), dim=1)
        x2 = local_operator(x1t, k=30)
        x2 = F.relu(self.conv2(x2))
        x2 = F.relu(self.conv21(x2))
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x2s, x2g = GDM(x2, M=256)

        y2s = self.SGCAM_2s(x2, x2s.transpose(2, 1))
        y2g = self.SGCAM_2g(x2, x2g.transpose(2, 1))
        z2 = torch.cat([y2s, y2g], 1)
        z2 = F.relu(self.conv22(z2))
        ###############
        x2t = torch.cat((x1t, z2), dim=1)
        x3 = local_operator(x2t, k=30)
        x3 = F.relu(self.conv3(x3))
        x3 = F.relu(self.conv31(x3))
        x3 = x3.max(dim=-1, keepdim=False)[0]
        z3 = F.relu(self.conv32(x3))
        ###############
        x = torch.cat((z1, z2, z3), dim=1)
        x = F.relu(self.conv4(x))
        x11 = F.adaptive_max_pool1d(x, 1).view(B, -1)
        x22 = F.adaptive_avg_pool1d(x, 1).view(B, -1)
        x = torch.cat((x11, x22), 1)

        # print(x.shape)
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x
    

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx, pairwise_distance


def local_operator(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx, _ = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    neighbor = x.view(batch_size * num_points, -1)[idx, :]

    neighbor = neighbor.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((neighbor-x, neighbor), dim=3).permute(0, 3, 1, 2)  # local and global all in

    return feature


def local_operator_withnorm(x, norm_plt, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    norm_plt = norm_plt.view(batch_size, -1, num_points)
    idx, _ = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    norm_plt = norm_plt.transpose(2, 1).contiguous()

    neighbor = x.view(batch_size * num_points, -1)[idx, :]
    neighbor_norm = norm_plt.view(batch_size * num_points, -1)[idx, :]

    neighbor = neighbor.view(batch_size, num_points, k, num_dims)
    neighbor_norm = neighbor_norm.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((neighbor-x, neighbor, neighbor_norm), dim=3).permute(0, 3, 1, 2)  # 3c

    return feature


def GDM(x, M):
    """
    Geometry-Disentangle Module
    M: number of disentangled points in both sharp and gentle variation components
    """
    k = 64  # number of neighbors to decide the range of j in Eq.(5)
    tau = 0.2  # threshold in Eq.(2)
    sigma = 2  # parameters of f (Gaussian function in Eq.(2))
    ###############
    """Graph Construction:"""
    device = torch.device('cuda')
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    idx, p = knn(x, k=k)  # p: -[(x1-x2)^2+...]

    # here we add a tau
    p1 = torch.abs(p)
    p1 = torch.sqrt(p1)
    mask = p1 < tau

    # here we add a sigma
    p = p / (sigma * sigma)
    w = torch.exp(p)  # b,n,n
    w = torch.mul(mask.float(), w)

    b = 1/torch.sum(w, dim=1)
    b = b.reshape(batch_size, num_points, 1).repeat(1, 1, num_points)
    c = torch.eye(num_points, num_points, device=device)
    c = c.expand(batch_size, num_points, num_points)
    D = b * c  # b,n,n

    A = torch.matmul(D, w)  # normalized adjacency matrix A_hat

    # Get Aij in a local area:
    idx2 = idx.view(batch_size * num_points, -1)
    idx_base2 = torch.arange(0, batch_size * num_points, device=device).view(-1, 1) * num_points
    idx2 = idx2 + idx_base2

    idx2 = idx2.reshape(batch_size * num_points, k)[:, 1:k]
    idx2 = idx2.reshape(batch_size * num_points * (k - 1))
    idx2 = idx2.view(-1)

    A = A.view(-1)
    A = A[idx2].reshape(batch_size, num_points, k - 1)  # Aij: b,n,k
    ###############
    """Disentangling Point Clouds into Sharp(xs) and Gentle(xg) Variation Components:"""
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.reshape(batch_size * num_points, k)[:, 1:k]
    idx = idx.reshape(batch_size * num_points * (k - 1))

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # b,n,c
    neighbor = x.view(batch_size * num_points, -1)[idx, :]
    neighbor = neighbor.view(batch_size, num_points, k - 1, num_dims)  # b,n,k,c
    A = A.reshape(batch_size, num_points, k - 1, 1)  # b,n,k,1
    n = A.mul(neighbor)  # b,n,k,c
    n = torch.sum(n, dim=2)  # b,n,c

    pai = torch.norm(x - n, dim=-1).pow(2)  # Eq.(5)
    pais = pai.topk(k=M, dim=-1)[1]  # first M points as the sharp variation component
    paig = (-pai).topk(k=M, dim=-1)[1]  # last M points as the gentle variation component

    pai_base = torch.arange(0, batch_size, device=device).view(-1, 1) * num_points
    indices = (pais + pai_base).view(-1)
    indiceg = (paig + pai_base).view(-1)

    xs = x.view(batch_size * num_points, -1)[indices, :]
    xg = x.view(batch_size * num_points, -1)[indiceg, :]

    xs = xs.view(batch_size, M, -1)  # b,M,c
    xg = xg.view(batch_size, M, -1)  # b,M,c

    return xs, xg


class SGCAM(nn.Module):
    """Sharp-Gentle Complementary Attention Module:"""
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(SGCAM, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv1d
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x, x_2):
        batch_size = x.size(0)

        g_x = self.g(x_2).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x_2).view(batch_size, self.inter_channels, -1)
        W = torch.matmul(theta_x, phi_x)  # Attention Matrix
        N = W.size(-1)
        W_div_C = W / N

        y = torch.matmul(W_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        y = W_y + x

        return y
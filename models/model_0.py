"""
resconv3.16
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .curvenet_util import *
def knn(x, k):
    k = k + 1
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
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
        points: [B, N, Dim]
        idx:    [B, S]
    Return:
        new_points:, indexed points data, [B, S, Dim]
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
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
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
def sample_and_group(npoint, radius, k, xyz, points, knn=True):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, D]
    """
    fps_idx=farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        groupe_idx = dists.argsort()[:, :, :k]  # B x npoint x K
    else:
        groupe_idx = query_ball_point(radius, k, xyz, new_xyz)
    new_grouped_points = index_points(points, groupe_idx) # [B, npoint, nsample, D]
    return new_xyz, new_grouped_points, new_points

def group_no_sample(points, k, idx=None):
    if idx is None:
        idx = knn(points, k=k)[:,:,:k]  # (batch_size, num_points, k)
    grouped_feature = index_points(points.permute(0,2,1), idx)
    return grouped_feature, idx

def get_act(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    else:
        return nn.ReLU(inplace=True)
class Conv1dBR(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, activation='leakyrelu'):
        super(Conv1dBR, self).__init__()
        if activation is not None:
            self.act = get_act(activation)
            self.layers = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias),
                nn.BatchNorm1d(out_channels),
                self.act)
        else:
            self.layers = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        return self.layers(x)
class LinearBR(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, activation='leakyrelu', Dropout=True):
        super(Conv1dBR, self).__init__()
        self.in_channels = in_channels
        self.act = get_act(activation)
        if Dropout:
            self.layers = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act, 
            nn.Dropout(0.5))
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=bias),
                nn.BatchNorm1d(out_channels),
                self.act
            )
    def forward(self, x):
        if x.shape[-2] == self.in_channels:
            for i, layer in enumerate(self.layers): 
                x = layer(x.permute(0, 2, 1)).permute(0, 2, 1) if i == 0 else layer(x)
        else:
            self.layers(x)
        return x       
class ResConv1dBR(nn.Module):
    def __init__(self, in_channels, res_expansion=1.0, bias=False, activation='relu'):
        super(ResConv1dBR, self).__init__()
        self.act = get_act(activation)
        self.net1 = Conv1dBR(in_channels, int(in_channels * res_expansion), bias=bias, activation=activation)
        self.net2 = Conv1dBR(int(in_channels * res_expansion), in_channels , bias=bias, activation=None)
    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)
class LocalExtraction(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=2, k =20, res_expansion=1,bias=True,activation='relu'):
        super(LocalExtraction, self).__init__()
        self.transfer = Conv1dBR(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ResConv1dBR(out_channels, res_expansion=res_expansion, bias=bias,activation=activation))
        self.operation = nn.Sequential(*operation) # TODO : 循环写成包便于调用

    def forward(self, x):
        b,num,k,dim=x.shape
        # x [B, npoint, nsample, D]->[B, D, nsample]
        x = x.view(-1, k, dim).permute(0,2,1) # [B*npoint, D, k]
        x = self.transfer(x) # TODO : GN
        x = self.operation(x)  # [B, D, k]

        x = F.adaptive_max_pool1d(x, 1)
        x = x.view(b, num, -1).permute(0, 2, 1)
        return x

class GlobalExtraction(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=2, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        """
        super(GlobalExtraction, self).__init__()
        self.transfer = Conv1dBR(in_channels, out_channels, bias=bias, activation=activation)

        operation = []
        for _ in range(blocks):
            operation.append(
                ResConv1dBR(out_channels, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(self.transfer(x))

class xyz_normal(nn.Module):
    def __init__(self,b,dim,num):#[b , dim, num]
        super(xyz_normal, self).__init__()
        self.b,self.d,self.num=b,dim,num
        self.ln = nn.LayerNorm(dim*num, elementwise_affine=False)
        
    def forward(self, x):#[b , dim, num]
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x = x.view(self.b,-1)
        x = self.ln(x).view(self.b,self.d,self.num)
        return x

class neighbor_normal(nn.Module):
    def __init__(self, in_channels,out_channels,bias=False):#[b , num,dim, k]
        super(neighbor_normal, self).__init__()
        self.conv0 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, out_channels]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, out_channels]))
        
    def forward(self, grouped_points, points):#[b,num,k,dim]  [b, num, dim]
        # b, _, num_points = points.shape
        b, num, k,dim= grouped_points.shape
        points = points.view(b, num, 1, -1).expand(-1, -1, k, -1)  

        res = grouped_points-points
        res = self.conv0(res.view(b*num,k,dim).permute(0,2,1)).permute(0,2,1).contiguous().view(b,num,k,-1)

        std = torch.std(res.view(b,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
        grouped_points = res/(std + 1e-5) # TODO :似乎要好好研究一下 GN
        grouped_points = self.affine_alpha*grouped_points + self.affine_beta
        return grouped_points,points#[b , num,k,dim]
class neighbor_normal1(nn.Module):
    def __init__(self, in_channels,num ,k):#[b , num,dim, k]
        super(neighbor_normal1, self).__init__()
        self.num,self.k=num,k

        # self.conv0 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
       
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, in_channels]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, in_channels]))
        
    def forward(self, grouped_points, points):#[b,num,k,dim]  [b, num, dim]
        # b, _, num_points = points.shape
        b, num, k,dim= grouped_points.shape
        points = points.view(b, self.num, 1, -1).expand(-1, -1, self.k, -1)  
        # print(points.shape)
        # os._exit(0)
        res = grouped_points-points

        # res = self.conv0(res.view(b*num,k,dim).permute(0,2,1)).permute(0,2,1).contiguous().view(b,num,k,-1)
        std = torch.std(res.view(b,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
        grouped_points = res/(std + 1e-5) # TODO :似乎要好好研究一下 GN
        grouped_points = self.affine_alpha*grouped_points + self.affine_beta
        return grouped_points,points#[b , num,k,dim]
class Input_embedding(nn.Module):  
    def __init__(self, in_channels, out_channels,k, mlp_num, bias=False,activation='leakyrelu'):
        super(Input_embedding, self).__init__()
        self.k = k
        out_channels=out_channels//2
        # self.xyznormal = xyz_normal(batch_size ,in_channels, npoint)# TODO 测试xyznorm
        self.neighbor_normal = neighbor_normal(in_channels,out_channels,bias=bias)

        self.conv = []
        
        for _ in range(mlp_num):
            self.conv.append(Conv1dBR(in_channels, out_channels, bias=bias, activation=activation))
            in_channels = out_channels
        self.conv = nn.Sequential(*self.conv) 

    def forward(self, xyz, idx=None):
        x = self.conv(xyz) # (b, dim, num)
        grouped_xyz, idx = group_no_sample(xyz, self.k, idx=idx) # (b, num, k, 3)

        grouped_points, _ = self.neighbor_normal(grouped_xyz, xyz.permute(0,2,1))
        grouped_points = grouped_points.permute(0,3,1,2).max(dim=-1, keepdim=False)[0]

        point_feature = torch.cat((x, grouped_points), dim=-2)
        return point_feature, idx
class SGPool(nn.Module):
    def __init__(self, npoint, radius, k, in_channels, bias=True, activation='leakyrelu'):
        super(SGPool, self).__init__()
        self.npoint, self.radius, self.k, self.in_channels= npoint, radius, k, in_channels
        # self.conv = Conv1dBR(in_channels, in_channels, bias, activation)
        # self.resconv = LocalExtraction(in_channels*2, in_channels, blocks=2, k=k, res_expansion=1,bias=bias,activation=activation)
        # self.neighbor_normal = neighbor_normal(in_channels, 32, npoint ,k)
    def forward(self, xyz, features):
        fps_idx=farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx).transpose(1, 2)
        new_features = index_points(features, fps_idx).transpose(1, 2)
        # sub_xyz, neighbor_features, new_points = \
        #     sample_and_group(self.npoint, self.radius, self.k, xyz, features) # [B, npoint, nsample, D]
        # # print(neighbor_features.shape)
        # # print(new_points.shape)
        # # 
        # grouped_points,points = self.neighbor_normal(neighbor_features, new_points)
        # # os._exit(0)
        # feature = torch.cat((grouped_points, points), dim=-1).permute(0, 1, 3, 2).contiguous()


        # neighbor_features = self.resconv(feature) # [B*npoint, D, nsample]
        # neighbor_features = neighbor_features\
        #     .view(-1, self.npoint, self.in_channels, self.k).permute(0, 2, 1, 3).contiguous()

        # sub_features = F.max_pool2d(neighbor_features, kernel_size=[1, self.k])  # bs, c, n, 1
        # sub_features = torch.squeeze(sub_features, -1)  # bs, c, n
        return new_xyz,new_features
class LPFA(nn.Module):
    def __init__(self, in_channels, out_channel,npoint, k, mlp_num=2):
        super(LPFA, self).__init__()
        self.k = k
        self.device = torch.device('cuda')
        self.xyz2feature = nn.Sequential(
                    nn.Conv2d(9, in_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_channels))

        in_channels = in_channels *2
        self.mlp = []
        for _ in range(mlp_num):
            self.mlp.append(nn.Sequential(nn.Conv2d(in_channels, out_channel, 1, bias=False),
                                 nn.BatchNorm2d(out_channel),
                                 nn.LeakyReLU(0.2)))
            in_channels = out_channel
        self.mlp = nn.Sequential(*self.mlp)     

        self.neighbor_normal = neighbor_normal(in_channels, 32, npoint ,k)

        # self.affine_alpha = nn.Parameter(torch.ones([1,1,1,in_channel]))
        # self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1,in_channel]))   

    def forward(self, x, xyz, idx=None): # x [b, d, n] xyz [b,3,n]
        x = self.group_feature(x, xyz, idx)
        x = self.mlp(x) # TODO :local

        # if self.initial:
        x = x.max(dim=-1, keepdim=False)[0]
        # else:
        # TODO : 测试最大池化
        # x = x.mean(dim=-1, keepdim=False)
        return x

    def group_feature(self, x, xyz, idx):
        batch_size, num_dims, num_points = x.size()

        if idx is None:
            idx = knn(xyz, k=self.k)[:,:,:self.k]  # (batch_size, num_points, k)

        idx_base = torch.arange(0, batch_size, device=self.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        # xyz = xyz.transpose(2, 1).contiguous() # bs, n, 3
        # point_feature = xyz.view(batch_size * num_points, -1)[idx, :]
        # point_feature = point_feature.view(batch_size, num_points, self.k, -1)  # bs, n, k, 3
        # points = xyz.view(batch_size, num_points, 1, 3).expand(-1, -1, self.k, -1)  # bs, n, k, 3

        # point_feature = torch.cat((points, point_feature, point_feature - points),
        #                         dim=3).permute(0, 3, 1, 2).contiguous()


        x = x.transpose(2, 1).contiguous() # bs, n, d
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)  #bs, n, k, d
        
        grouped_points,points = self.neighbor_normal(feature, x)
        
        # x = x.view(batch_size, num_points, 1, num_dims)
        # # edge_feature = feature - x

        # # ----------------------------------------------------------------
        # std = torch.std((feature-x).reshape(batch_size,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
        # # print(std.shape)
        # # print(std)
        # grouped_points = (feature-x)/(std + 1e-5)
        # grouped_points = self.affine_alpha*grouped_points + self.affine_beta
        # os._exit(0)
        feature = torch.cat((grouped_points, points), dim=-1).permute(0, 3, 1, 2).contiguous()
        # TODO :边信息和节点信息是否需要归一化
        # TODO :是否需要带上point_feature
        #point_feature = torch.cat((feature - x, feature, point_feature), dim=-1)

        # edge_feature = edge_feature.permute(0, 3, 1, 2).contiguous()
        # point_feature = self.xyz2feature(point_feature)  #bs, c, n, k

        # feature = F.leaky_relu(edge_feature + point_feature, 0.2)
        return feature #bs, c, n, k

class GGraph(nn.Module):
    def __init__(self, npoint, radius, k, in_channels, bias=True, activation='leakyrelu'):
        super(GGraph, self).__init__()
        self.k,self.npoint, self.radius,  self.in_channels=k, npoint, radius, in_channels
        # self.conv = Conv1dBR(in_channels, in_channels, bias, activation)
        # self.resconv = LocalExtraction(in_channels*2, in_channels, blocks=2, k=k, res_expansion=1,bias=bias,activation=activation)
        self.neighbor_normal = neighbor_normal1(in_channels, npoint ,k)
    def forward(self, xyz, features,idx = None):
        # grouped_xyz,_ = group_no_sample(xyz, self.k, idx=idx) # (b, num, k, dim)
        if idx is None:
            idx = knn(xyz, self.k)[:,:,:self.k] # 静态图
        # idx = knn(features, self.k)[:,:,:self.k] # 动态图
        grouped_feature = index_points(features.permute(0,2,1), idx) # (b, num, k, dim)
        # print(grouped_feature.shape)
        grouped_feature, features = self.neighbor_normal(grouped_feature, features)
        feature = torch.cat((grouped_feature, features), dim=-1)
        # print(feature.shape)
        return feature# (b, num, k, dim)

class BackboneFeature(nn.Module):
    def __init__(self, npoint, in_channels, output_channels, radius, k, bottleneck_ratio=2, mlp_num=2, activation='relu'):
        super(BackboneFeature, self).__init__()
    
        self.in_channels,self.npoint,self.k = in_channels,npoint,k
        self.output_channels = output_channels
        planes = in_channels // bottleneck_ratio
        self.sgpool = SGPool(npoint, radius, k, in_channels, bias=False, activation=activation)
        self.ggraph = GGraph(npoint, radius, k, in_channels, bias=False, activation='leakyrelu')

        self.Localextraction = LocalExtraction(in_channels*2, planes, blocks=2, k =20, res_expansion=1,bias=False,activation='relu')


        self.globalExtraction = GlobalExtraction(planes, planes, blocks=2, res_expansion=1, bias=False, activation=activation)

        # self.lpfa = LPFA(planes, planes, npoint=npoint, k=k, mlp_num=mlp_num)

        self.conv2 = Conv1dBR(planes, output_channels, bias=False, activation=None)

        if in_channels != output_channels:
            self.shortcut = Conv1dBR(in_channels, output_channels, bias=False, activation=None)
        self.relu = get_act(activation)
        

    def forward(self, xyz, x,idx= None):
        # max pool
        if xyz.size(-1) != self.npoint: #(b,d,n)
            xyz, x = self.sgpool(xyz.transpose(1, 2), x.transpose(1, 2))# bs, d, n # TODO :取代最远点采样
        shortcut = x

        x = self.ggraph(xyz, x, idx) # 生成图 # (b, num, k, dim)
        x = self.Localextraction(x) #
        x = self.globalExtraction(x)

        x = self.conv2(x)  # bs, c, n
        # os._exit(1)
        if self.in_channels != self.output_channels:
            shortcut = self.shortcut(shortcut)

        x = self.relu(x + shortcut)
        return xyz, x

class Module(nn.Module):
    def __init__(self, cfg=None, num_classes=40, k=20):
        super(Module, self).__init__()
        self.input_embedding = Input_embedding(in_channels=3, out_channels=64, k=k, mlp_num=1)
        
        # self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)
        # encoder
        self.cic11 = BackboneFeature(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1)
        # self.cic12 = BackboneFeature(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, mlp_num=1)
        
        self.cic21 = BackboneFeature(npoint=1024, radius=0.05, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, mlp_num=1)
        # self.cic22 = BackboneFeature(npoint=1024, radius=0.1, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, mlp_num=1)

        self.cic31 = BackboneFeature(npoint=256, radius=0.1, k=k, in_channels=256, output_channels=256, bottleneck_ratio=2, mlp_num=1)
        # self.cic32 = BackboneFeature(npoint=256, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, mlp_num=1)

        self.cic41 = BackboneFeature(npoint=256, radius=0.2, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2, mlp_num=1)
        # self.cic42 = BackboneFeature(npoint=64, radius=0.4, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4, mlp_num=1)

        
        # self.conv0 = nn.Sequential(
        #     nn.Conv1d(512, 1024, kernel_size=1, bias=False),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True))
        self.classification = nn.Sequential(
            nn.Linear(512*2, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), # inplace = True,会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
            )
        # self.conv1 = nn.Linear(1024 * 2, 512, bias=False)
        # self.conv2 = nn.Linear(512, num_classes)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=0.5)

    def forward(self, xyz):
        # print(xyz.shape)
        xyz = xyz.permute(0,2,1).contiguous()
        points ,idx= self.input_embedding(xyz)

        l1_xyz, l1_points = self.cic11(xyz, points,idx)
        # l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        # l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        # l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)
 
        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        # l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        # x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(l4_points, 1)
        x_avg = F.adaptive_avg_pool1d(l4_points, 1)
        
        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        # print(x.shape)
        # os._exit(0)
        x = self.classification(x)
        return x



if __name__ == '__main__':
    import sys
    sys.path.append("../") 
    from utils import read_yaml
    cfg = read_yaml('../configs/config.yaml')
    data = torch.rand(32, 1024, 3)
    print("===> testing pointMLP ...")
    model = Module(cfg).cuda()
    out = model(data.cuda())
    print(out.shape)
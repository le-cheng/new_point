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
def sample_and_group(npoint, radius, nsample, xyz, points, knn=True):
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
    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        groupe_idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    else:
        groupe_idx = query_ball_point(radius, nsample, xyz, new_xyz)
    new_grouped_points = index_points(points, groupe_idx) # [B, npoint, nsample, D]
    return new_xyz, new_grouped_points, groupe_idx
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
        """
        input: [b,g,k,d]: output:[b,d,g]
        """
        self.in_channels, self.k = in_channels, k
        self.transfer = Conv1dBR(in_channels, out_channels, bias=bias, activation=activation)
        
        operation = []
        for _ in range(blocks):
            operation.append(
                ResConv1dBR(out_channels, res_expansion=res_expansion, bias=bias,activation=activation))
        self.operation = nn.Sequential(*operation) # TODO : 循环写成包便于调用

    def forward(self, x):
        # x [B, npoint, nsample, D]->[B, D, nsample]
        x = x.permute(0, 1, 3, 2).contiguous().view(-1, self.in_channels, self.k) # [B*npoint, D, nsample]
        x = self.transfer(x)
        x = self.operation(x)  # [B, D, nsample]
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

class SGPool(nn.Module):
    def __init__(self, npoint, radius, k, in_channels, bias=True, activation='leakyrelu'):
        super(SGPool, self).__init__()
        self.npoint, self.radius, self.k, self.in_channels= npoint, radius, k, in_channels
        # self.conv = Conv1dBR(in_channels, in_channels, bias, activation)
        self.resconv = LocalExtraction(in_channels, in_channels, blocks=2, k=k, res_expansion=1,bias=bias,activation=activation)

    def forward(self, xyz, features):
        sub_xyz, neighbor_features, _ = \
            sample_and_group(self.npoint, self.radius, self.k, xyz, features) # [B, npoint, nsample, D]
        neighbor_features = self.resconv(neighbor_features) # [B*npoint, D, nsample]
        neighbor_features = neighbor_features\
            .view(-1, self.npoint, self.in_channels, self.k).permute(0, 2, 1, 3).contiguous()

        sub_features = F.max_pool2d(neighbor_features, kernel_size=[1, self.k])  # bs, c, n, 1
        sub_features = torch.squeeze(sub_features, -1)  # bs, c, n
        return sub_xyz.transpose(1, 2), sub_features

class LPFA1(nn.Module):
    def __init__(self, in_channel, out_channel, k, mlp_num=2, initial=False):
        super(LPFA, self).__init__()
        self.k = k
        self.device = torch.device('cuda')
        self.initial = initial

        if not initial:
            self.xyz2feature = nn.Sequential(
                        nn.Conv2d(9, in_channel, kernel_size=1, bias=False),
                        nn.BatchNorm2d(in_channel))

        self.mlp = []
        for _ in range(mlp_num):
            self.mlp.append(nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, bias=False),
                                 nn.BatchNorm2d(out_channel),
                                 nn.LeakyReLU(0.2)))
            in_channel = out_channel
        self.mlp = nn.Sequential(*self.mlp)        

    def forward(self, x, xyz, idx=None):
        x = self.group_feature(x, xyz, idx)
        x = self.mlp(x) # TODO :local

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
class LPFA(nn.Module):
    def __init__(self, in_channel, out_channel, k, mlp_num=2):
        super(LPFA, self).__init__()
        self.k = k
        self.device = torch.device('cuda')

        
        self.xyz2feature = nn.Sequential(
                    nn.Conv2d(9, in_channel, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_channel))

        in_channel = in_channel *2
        self.mlp = []
        for _ in range(mlp_num):
            self.mlp.append(nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, bias=False),
                                 nn.BatchNorm2d(out_channel),
                                 nn.LeakyReLU(0.2)))
            in_channel = out_channel
        self.mlp = nn.Sequential(*self.mlp)        

    def forward(self, x, xyz, idx=None): # x [b, d, n] xyz [b,3,n]
        x = self.group_feature(x, xyz, idx)
        x = self.mlp(x) # TODO :local

        # if self.initial:
        #     x = x.max(dim=-1, keepdim=False)[0]
        # else:
        # TODO : 测试最大池化
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

        x = x.transpose(2, 1).contiguous() # bs, n, d
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)  #bs, n, k, d
        x = x.view(batch_size, num_points, 1, num_dims)
        # edge_feature = feature - x

        feature = torch.cat((x-feature, feature), dim=-1).permute(0, 3, 1, 2).contiguous()
        # TODO :边信息和节点信息是否需要归一化
        # TODO :是否需要带上point_feature
        #point_feature = torch.cat((feature - x, feature, point_feature), dim=-1)

        # edge_feature = edge_feature.permute(0, 3, 1, 2).contiguous()
        # point_feature = self.xyz2feature(point_feature)  #bs, c, n, k

        # feature = F.leaky_relu(edge_feature + point_feature, 0.2)
        return feature #bs, c, n, k
class CIC(nn.Module):
    def __init__(self, npoint, in_channels, output_channels, radius, k, bottleneck_ratio=2, mlp_num=2, activation='leakyrelu'):
        super(CIC, self).__init__()
        self.in_channels,self.npoint,self.k = in_channels,npoint,k
        self.output_channels = output_channels
        planes = in_channels // bottleneck_ratio
        self.SGPool = SGPool(npoint, radius, k, in_channels, bias=False, activation=activation)
        self.conv1 = GlobalExtraction(in_channels, planes, blocks=2, res_expansion=1, bias=False, activation=activation)

        self.lpfa = LPFA(planes, planes, k, mlp_num=mlp_num)

        self.conv2 = Conv1dBR(planes, output_channels, bias=False, activation=None)

        if in_channels != output_channels:
            self.shortcut = Conv1dBR(in_channels, output_channels, bias=False, activation=None)
        self.relu = get_act(activation)
        

    def forward(self, xyz, x):
        # max pool
        if xyz.size(-1) != self.npoint:#(b,d,n)
            xyz, x = self.SGPool(xyz.transpose(1, 2).contiguous(), x.transpose(1, 2).contiguous())# bs, d, n
        shortcut = x
        x = self.conv1(x)  # bs, d', n

        idx = knn(xyz, self.k)[:,:,:self.k] # 静态图

        x = self.lpfa(x, xyz, idx=idx)#bs, c', n, k
        # print(x.shape)
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
        self.cic11 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=2, mlp_num=1)
        self.cic12 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, mlp_num=1)
        
        self.cic21 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1)
        self.cic22 = CIC(npoint=1024, radius=0.1, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, mlp_num=1)

        self.cic31 = CIC(npoint=256, radius=0.1, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, mlp_num=1)
        self.cic32 = CIC(npoint=256, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, mlp_num=1)

        self.cic41 = CIC(npoint=64, radius=0.2, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2, mlp_num=1)
        self.cic42 = CIC(npoint=64, radius=0.4, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4, mlp_num=1)

        
        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))
        self.classification = nn.Sequential(
            nn.Linear(1024 * 2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), # inplace = True,会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
            )
        self.conv1 = nn.Linear(1024 * 2, 512, bias=False)
        self.conv2 = nn.Linear(512, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)

    def forward(self, xyz):
        xyz = xyz.permute(0,2,1).contiguous()
        points = self.input_embedding(xyz)

        l1_xyz, l1_points = self.cic11(xyz, points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)
 
        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)
        
        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        x = self.classification(x)
        return x

class Input_embedding(nn.Module):  
    def __init__(self, in_channels, out_channels, k, mlp_num, bias=False,activation='leakyrelu'):
        super(Input_embedding, self).__init__()
        self.k = k
        self.conv = []
        in_channels=in_channels*3
        for _ in range(mlp_num):
            self.conv.append(Conv1dBR(in_channels, out_channels, bias=bias, activation=activation))
            in_channels = out_channels
        self.conv = nn.Sequential(*self.conv) 
        self.device = torch.device('cuda')

    def forward(self, xyz, idx=None):
        batch_size, _, num_points = xyz.shape
        # print(batch_size)
        # os._exit(0)
        if idx is None:
            idx = knn(xyz, k=self.k)[:,:,:self.k]  # (batch_size, num_points, k)
        
        
        idx_base = torch.arange(0, batch_size, device=self.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        xyz = xyz.transpose(2, 1).contiguous() # bs, n, 3
        point_feature = xyz.view(batch_size * num_points, -1)[idx, :]
        point_feature = point_feature.view(batch_size, num_points, self.k, -1)  # bs, n, k, 3
        points = xyz.view(batch_size, num_points, 1, -1).expand(-1, -1, self.k, -1)  # bs, n, k, 3

        point_feature = torch.cat((
            points, point_feature, points - point_feature # TODO :距离，数据标准化
            ), dim=3).permute(0, 1, 3, 2).contiguous().view(batch_size* num_points, -1, self.k)
        
        # x = self.group_feature(x, xyz, idx)
        x = self.conv(point_feature).view(batch_size, num_points, -1, self.k).permute(0,2,1,3)
        x = x.max(dim=-1, keepdim=False)[0]
        return x

if __name__ == '__main__':
    data = torch.rand(16, 1024, 3)
    print("===> testing pointMLP ...")
    model = Module().cuda()
    out = model(data.cuda())
    print(out.shape)